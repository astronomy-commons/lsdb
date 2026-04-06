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

    FROM python:3.12-slim

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
      minimum: 1
      maximum: 6

Apply the manifest:

.. code-block:: bash

    kubectl apply -f dask-cluster.yaml

The PersistentVolumeClaim ``hats-catalog-pvc`` should point to storage containing your
HATS catalogs. If your catalogs are accessed over the network (e.g. via
``https://data.lsdb.io``), you can remove the volume mount entirely.

Connecting LSDB to the cluster
------------------------------

Once the scheduler is running, connect to it from a Python session inside the cluster
(e.g. a Jupyter pod in the same namespace) or via ``kubectl port-forward``.

From inside the cluster:

.. code-block:: python

    from dask.distributed import Client
    import lsdb

    client = Client("tcp://lsdb-cluster-scheduler:8786")

    catalog = lsdb.open_catalog("/data/catalogs/my_catalog")
    result = catalog.cone_search(ra=180, dec=0, radius_arcsec=600).compute()
    client.close()

From outside the cluster, forward the scheduler port first:

.. code-block:: bash

    kubectl port-forward svc/lsdb-cluster-scheduler 8786:8786

Then connect to ``tcp://localhost:8786`` in your local Python session.

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

Tasks are slow or idle workers
..............................

If you see many idle workers while tasks queue up, the graph may not have enough partitions
to keep all workers busy.

- Check ``catalog.npartitions`` to see how many partitions are available.
- Use ``lsdb.open_catalog(search_filter=...)`` to narrow the spatial region before
  expanding to the full sky.

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
