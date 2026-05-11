Kubernetes deployment with the Dask Operator
============================================

For workloads that exceed what a single machine or a traditional HPC cluster can offer,
Kubernetes provides a flexible platform for running LSDB at scale.
A Kubernetes cluster can dynamically allocate resources, scale workers up and down based on demand,
and provide a reproducible environment for data processing pipelines.

This guide walks through deploying a `Dask Kubernetes Operator <https://kubernetes.dask.org/en/latest/operator.html>`__
cluster suitable for LSDB workloads.
It complements the :doc:`Dask cluster configuration tips </tutorials/dask-cluster-tips>` page,
which covers local and SLURM-based setups.

.. note::

    Kubernetes adds operational complexity compared to a local cluster or SLURM job.
    If your data fits in memory on a single machine, a
    `LocalCluster <https://distributed.dask.org/en/stable/api.html#distributed.LocalCluster>`__
    is simpler and often sufficient.
    Consider Kubernetes when you need multi-node scaling, autoscaling, or a shared
    platform that multiple users or pipelines can share.

Prerequisites
-------------

Before starting, you will need:

- A running Kubernetes cluster (v1.25+) with ``kubectl`` configured
- `Helm <https://helm.sh/docs/intro/install/>`__ v3 installed
- A container registry you can push images to (Docker Hub, GitHub Container Registry, etc.)
- Basic familiarity with Kubernetes concepts (pods, services, persistent volumes)

Building a container image
--------------------------

The Dask project does not ship an official image with LSDB pre-installed.
Instead, you build your own image with the packages your workflow requires.
This keeps the image minimal and lets you pin exact versions.

A minimal ``Dockerfile``:

.. code-block:: dockerfile

    FROM python:3.12.3-slim

    RUN pip install --no-cache-dir \
        lsdb \
        dask[complete] \
        distributed

    # Add any additional packages your workflow needs, for example:
    # RUN pip install --no-cache-dir s3fs jupyter

Build and push the image to your registry:

.. code-block:: bash

    docker build -t registry.example.com/lsdb-dask:latest .
    docker push registry.example.com/lsdb-dask:latest

.. tip::

    Pin package versions in production images (e.g. ``lsdb==0.5.0``) so that
    your results are reproducible across runs.

.. tip::

    Pin the Python base image to a specific patch release (``python:3.12.3-slim``),
    not just the minor series (``python:3.12-slim``), and match it to the Python
    version of the client environment you connect from. Dask compares Python and
    package versions across the client, scheduler, and workers, and minor
    differences can produce a ``VersionMismatchWarning`` or, in some cases,
    serialization errors.

Installing the Dask Kubernetes Operator
---------------------------------------

The `Dask Kubernetes Operator <https://kubernetes.dask.org/en/latest/operator.html>`__
manages ``DaskCluster`` custom resources, handling scheduler and worker pod lifecycles
and optional autoscaling.

Install the operator with Helm:

.. code-block:: bash

    helm repo add dask https://helm.dask.org
    helm repo update

    helm install dask-operator dask/dask-kubernetes-operator \
        --namespace dask-operator \
        --create-namespace

.. note::

    The operator is cluster-scoped. You only need one installation per cluster.
    If another team has already installed it, check with
    ``kubectl get crd | grep dask`` and skip this step if the CRDs exist.

Verify that the operator pod is running:

.. code-block:: bash

    kubectl get pods -n dask-operator

Creating a DaskCluster
----------------------

The following manifest creates a Dask cluster with a scheduler, workers, and an autoscaler.
Adjust resource requests, replica counts, and the image reference to match your environment.

.. code-block:: yaml

    apiVersion: kubernetes.dask.org/v1
    kind: DaskCluster
    metadata:
      name: lsdb-cluster
    spec:
      worker:
        replicas: 2
        spec:
          containers:
            - name: worker
              image: "registry.example.com/lsdb-dask:latest"
              args:
                - dask-worker
                - --name
                - $(DASK_WORKER_NAME)
                - --nthreads
                - "2"
              resources:
                requests:
                  cpu: "2"
                  memory: "4Gi"
                limits:
                  memory: "4Gi"
              volumeMounts:
                - name: catalog-data
                  mountPath: /data
          volumes:
            - name: catalog-data
              persistentVolumeClaim:
                claimName: hats-catalog-pvc
      scheduler:
        spec:
          containers:
            - name: scheduler
              image: "registry.example.com/lsdb-dask:latest"
              args:
                - dask-scheduler
              resources:
                requests:
                  cpu: "2"
                  memory: "6Gi"
                limits:
                  memory: "6Gi"
              ports:
                - name: tcp-comm
                  containerPort: 8786
                  protocol: TCP
                - name: http-dashboard
                  containerPort: 8787
                  protocol: TCP
        service:
          type: ClusterIP
          selector:
            dask.org/cluster-name: lsdb-cluster
            dask.org/component: scheduler
          ports:
            - name: tcp-comm
              protocol: TCP
              port: 8786
              targetPort: tcp-comm
            - name: http-dashboard
              protocol: TCP
              port: 8787
              targetPort: http-dashboard
    ---
    apiVersion: kubernetes.dask.org/v1
    kind: DaskAutoscaler
    metadata:
      name: lsdb-cluster
    spec:
      cluster: lsdb-cluster
      minimum: 2
      maximum: 6

Apply the manifest:

.. code-block:: bash

    kubectl apply -f dask-cluster.yaml

.. tip::

    If your container image is in a private registry, add ``imagePullSecrets``
    to both the scheduler and worker pod specs. Public images (e.g. on Docker Hub
    or GitHub Container Registry with public visibility) do not need this.

.. tip::

    The example above sets ``limits.memory`` but not ``limits.cpu``, which lets
    workers burst above their CPU request when the node has spare capacity. If
    your namespace has a ``ResourceQuota`` or ``LimitRange`` that requires every
    container to declare ``limits.cpu``, the manifest will be rejected with
    ``failed quota: ... must specify limits.cpu``. Add ``cpu: "2"`` (or your
    chosen ceiling) under ``limits`` for both the worker and scheduler containers
    in that case.

The PersistentVolumeClaim ``hats-catalog-pvc`` should point to storage containing your
HATS catalogs. If your catalogs are accessed over the network (e.g. via
``https://data.lsdb.io``), you can remove the volume mount entirely.

Providing the catalog volume
............................

The manifest above mounts a PVC named ``hats-catalog-pvc`` at ``/data`` on every
worker. You need to create that PVC (and, depending on your environment, the
PersistentVolume backing it) before applying the ``DaskCluster`` manifest.

The simplest case is a cluster with a default ``StorageClass`` that supports
``ReadWriteMany`` (NFS, CephFS, EFS, Azure Files, GCP Filestore, etc.). All
workers must be able to read the catalog at the same time, so ``ReadWriteMany``
is required for multi-worker setups:

.. code-block:: yaml

    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: hats-catalog-pvc
    spec:
      accessModes:
        - ReadWriteMany
      resources:
        requests:
          storage: 100Gi
      # storageClassName: nfs-client   # set to a RWX-capable StorageClass

.. tip::

    If the PVC stays in ``Pending`` after you apply it, your cluster likely has
    no default ``StorageClass``, or its default does not support
    ``ReadWriteMany``. Run ``kubectl get storageclass`` to list what is
    available, then uncomment ``storageClassName`` and set it to the name of a
    RWX-capable class.

If your cluster does not have a dynamic provisioner for shared storage, you can
back the PVC with a statically defined NFS PersistentVolume. Adjust the server
address, export path, and capacity to match your environment:

.. code-block:: yaml

    apiVersion: v1
    kind: PersistentVolume
    metadata:
      name: hats-catalog-pv
    spec:
      capacity:
        storage: 100Gi
      accessModes:
        - ReadWriteMany
      persistentVolumeReclaimPolicy: Retain
      nfs:
        server: nfs.example.com
        path: /exports/hats-catalogs
      mountOptions:
        - nfsvers=4.1
    ---
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: hats-catalog-pvc
    spec:
      accessModes:
        - ReadWriteMany
      resources:
        requests:
          storage: 100Gi
      storageClassName: ""   # bind to the static PV above
      volumeName: hats-catalog-pv

Apply the storage objects before the ``DaskCluster``:

.. code-block:: bash

    kubectl apply -f hats-catalog-storage.yaml
    kubectl get pvc hats-catalog-pvc

Wait for the PVC to report ``Bound`` before applying the ``DaskCluster`` manifest;
worker pods will stay in ``Pending`` until the volume is available.

.. note::

    For object-storage catalogs (S3, GCS, Azure Blob), do not use a PVC at all.
    Drop the ``volumes`` and ``volumeMounts`` blocks from the worker spec, install
    the appropriate filesystem driver in your image (``s3fs``, ``gcsfs``,
    ``adlfs``), and pass the bucket URL to ``lsdb.open_catalog`` (e.g.
    ``s3://my-bucket/catalogs/my_catalog``).

Connecting LSDB to the cluster
------------------------------

Once the scheduler is running, connect to it from a Python session inside the cluster
(e.g. a Jupyter pod in the same namespace) or via ``kubectl port-forward``.
The two paths below are mutually exclusive; pick whichever matches where your Python
session runs and use that connection string consistently.

Option A: From inside the cluster
.................................

If your Python session runs in a pod in the same namespace as the scheduler (for
example, a Jupyter pod), connect directly to the scheduler service:

.. code-block:: python

    from dask.distributed import Client
    import lsdb

    client = Client("tcp://lsdb-cluster-scheduler:8786")

    # Use search_filter at open time to avoid loading the full catalog into memory.
    # Adding columns= further reduces the memory footprint per partition.
    catalog = lsdb.open_catalog(
        "/data/catalogs/my_catalog",
        search_filter=lsdb.ConeSearch(ra=180, dec=0, radius_arcsec=600),
        columns=["ra", "dec", "phot_g_mean_mag"],
    )
    result = catalog.compute()
    print(result.head())
    client.close()

The ``print(result.head())`` line gives you immediate, human-readable confirmation
that the cluster is functioning end to end (catalog open, distributed compute,
result collection).

Option B: From outside the cluster
..................................

If your Python session runs on your laptop or another machine outside the cluster,
forward the scheduler port first:

.. code-block:: bash

    kubectl port-forward svc/lsdb-cluster-scheduler 8786:8786

Then, in your local Python session, use ``tcp://localhost:8786`` instead of the
in-cluster service name when constructing the client:

.. code-block:: python

    client = Client("tcp://localhost:8786")

The rest of the LSDB code (``open_catalog``, ``compute``, ``print(result.head())``)
is identical to Option A.

Resource sizing recommendations
-------------------------------

The table below provides starting-point resource allocations based on catalog size.
These are guidelines; actual requirements depend on the complexity of your operations
(cross-matching is more memory-intensive than simple filtering).

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 35

   * - Catalog size
     - Workers
     - CPU / worker
     - Memory / worker
     - Notes
   * - < 10 GB
     - 2
     - 2
     - 4 Gi
     - A local cluster may be simpler
   * - 10 -- 100 GB
     - 4
     - 2
     - 4 Gi
     - Good baseline for cone searches and filtering
   * - 100 GB -- 1 TB
     - 6 -- 12
     - 2
     - 8 Gi
     - Cross-matching benefits from more workers
   * - > 1 TB
     - 12+
     - 4
     - 16 Gi
     - Consider dedicated nodes and higher thread counts

As discussed in :doc:`Dask cluster configuration tips </tutorials/dask-cluster-tips>`,
prefer more workers with fewer threads over fewer workers with many threads.
Keep ``--nthreads`` at 1 or 2 per worker for LSDB workloads.

Monitoring
----------

The Dask dashboard is invaluable for understanding task progress and diagnosing bottlenecks.
The scheduler exposes it on port 8787 by default.

Forward it locally:

.. code-block:: bash

    kubectl port-forward svc/lsdb-cluster-scheduler 8787:8787

Then open `<http://localhost:8787/status>`__ in your browser.
See the :ref:`Dask Dashboard <tutorials/dask-cluster-tips:Dask Dashboard>` section
for details on interpreting the dashboard panels.

If your cluster has an ingress controller, you can expose the dashboard through an
``Ingress`` resource instead.

Tips and troubleshooting
------------------------

Workers killed by OOM
.....................

If workers are killed unexpectedly, they are likely exceeding their memory limit.
Check the Dask dashboard memory bars or ``kubectl describe pod`` for ``OOMKilled`` events.

- Increase the memory limit and request for workers.
- Use ``lsdb.open_catalog(columns=...)`` to load only the columns you need.
- Reduce ``--nthreads`` to lower per-worker memory pressure.

Idle workers
............

If you see many idle workers while a small number stay busy, the graph likely does
not have enough partitions to spread the work across the cluster.

- Check ``catalog.npartitions``. As a rule of thumb, you want at least one
  partition per worker thread, and ideally several, so that the scheduler has
  something to assign to each worker.
- If ``npartitions`` is much smaller than your worker count, scale the cluster
  down to match the available parallelism. HATS catalogs are partitioned at
  import time by HEALPix order, so the partition count of an existing catalog
  is fixed; if you need more partitions, re-import the source data with a finer
  HEALPix order using the ``hats-import`` pipeline.
- Inspect the task stream and progress panels in the Dask dashboard to confirm
  that the bottleneck is the number of tasks, not data movement or memory
  pressure (which look very different in the dashboard).

Note that narrowing the spatial region with ``search_filter`` would *reduce* the
partition count further and make this symptom worse. ``search_filter`` is the right
tool for the slow-tasks-and-memory-pressure case below, not for idle workers.

Slow tasks or memory pressure
.............................

If individual tasks take a long time, workers spill to disk, or the dashboard
shows yellow or red memory bars, the per-partition working set is likely too
large for the worker's memory budget.

- Use ``lsdb.open_catalog(search_filter=...)`` to narrow the spatial region so
  each partition processes less data.
- Use ``lsdb.open_catalog(columns=[...])`` to read only the columns you need.
- Increase the worker memory request and limit in the ``DaskCluster`` manifest.
- Reduce ``--nthreads`` so each task on a worker has more memory headroom.

Scheduler unreachable
.....................

If the client cannot connect to the scheduler:

- Verify the scheduler pod is running: ``kubectl get pods -l dask.org/component=scheduler``
- Check that the service exists: ``kubectl get svc lsdb-cluster-scheduler``
- Ensure your client is in the same namespace or using the fully qualified service name
  (e.g. ``lsdb-cluster-scheduler.<namespace>.svc.cluster.local``).

Image version mismatch
......................

The scheduler and all workers must use the same image and package versions.
Version mismatches between the client environment and the cluster can cause
serialization errors.
Use a pinned image tag (not ``latest``) and ensure your local environment
matches the versions installed in the image.
