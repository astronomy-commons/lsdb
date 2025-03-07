Dask cluster configuration tips
===============================

LSDB uses the `Dask <https://dask.org/>`_ framework for parallel and out-of-memory computing.
Dask is a flexible library for parallel computing in Python that scales from single machines to large clusters.
When using LSDB, it is usually worth setting up a Dask cluster to take advantage of parallel computing.
With no Dask cluster, LSDB would use a single CPU core for the computations, which is prohibitive for large datasets.

Here, we provide some tips on how to set up a Dask cluster for LSDB computations. Note that
`Dask <https://dask.org/>`_ also provides it's own
`best practices <https://docs.dask.org/en/stable/best-practices.html>`_, which may also be useful to consult.

Local Cluster on Single Machine or Node
---------------------------------------

Even on a single machine, Dask recommends using their distributed client.
This simple example runs a local cluster and starts a client connected to it.

.. code-block:: python

    from dask.distributed import Client
    from lsdb import read_hats, ConeSearch

    catalog = lsdb.read_hats(
        'https://data.lsdb.io/hats/gaia_dr3/gaia',
        search_filter=ConeSearch(ra=0, dec=0, radius=1),
    )
    client = Client()
    df = catalog.compute()
    client.close()


In this example, Dask would typically allocate multiple workers, and set number of Python threads and
memory limit to match your machine configuration.
As you run more computationally intensive graphs, you will want to consider configuring the cluster to
to better distribute your machine's resources.
Usually the client initialization would look like this:

.. code-block:: python

    client = dask.distributed.Client(
        # Number of Dask workers - Python processes to run
        n_workers=16,
        # Limits number of Python threads per worker
        threads_per_worker=2,
        # Memory limit, per worker, which here is also 10 GB per worker thread
        memory_limit="20GB",
    )
    client


In general, increasing the number of workers executing tasks in your graph in parallel is more useful than
increasing the number of threads available for each worker. In your cluster configuration, this means that it is
more useful to increase the number of workers, ``n_workers``, while keeping ``threads_per_worker`` to a small
value (say 1 or 2 threads per worker).
This is because ``threads_per_worker`` determines the number of Dask tasks each worker can process concurrently.
That's why larger ``threads_per_worker`` values may lead to large memory consumption. Additionally, larger
numbers of threads can increase the Global Interpreter Lock (GIL) contention.

In terms of RAM allocation for each worker, we have found that returns diminish past 10 GB per each thread
allocated to that worker when processing LSDB workloads.
Usage of ``lsdb.read_hats(catalogs=..., filters=...)`` may make the memory footprint much smaller, which would
allow you to allocate less memory per worker, and thus use more workers and make the analysis run faster.

When executing custom code/functions through LSDB's interface, keep in mind that any intermediate products
created in that function affect the memory footprint of the overall workflow. For example, in the below
example our code copies a dataframe input effectively doubling the amount of input memory. Being aware of
the memory performance of your analysis code is highly encouraged for general LSDB/Dask performance, and
memory allocation may need to be increased accordingly.

This code gives an example of a pipeline which would duplicate the data and use more memory:

.. code-block:: python

    def my_func(df):
        df2 = df.copy()
        return df2.query("a>5")

    new_catalog = catalog.map_partitions(df)


Some workflows can have very unbalanced memory load, some just one or few tasks would use much more memory than others.
In such cases you may set total memory limit ``memory_limit * n_workers`` larger than the actual amount of
memory on your system.

Multiple Node Cluster
---------------------

With multiple nodes, you would usually have a scheduler running on one node and Dask workers being distributed across the nodes.
In this case each computational node would run one or more Dask workers, while each worker may take few Dask tasks (usually one per LSDB partition) and use multiple threads.

High-Performance Computing Cluster with SLURM
.............................................

Dask ecosystem has a `dask-jobqueue <https://jobqueue.dask.org/en/latest/>`_ package that allows to run Dask on HPC clusters.
It provides a way to submit Dask workers as SLURM jobs, and to scale the number of workers dynamically.
Unfortunately, ``dask-jobqueue`` does not support selecting both the number of worker cores and Dask threads per worker separately.
We found it may be a problem for some SLURM clusters that require to specify the exact number of cores and memory per job.

The following configuration is an example that was run on `PSC <https://www.psc.edu/>`_, and contains some
specific settings useful for its hardware of that cluster.
This configuration runs 60 SLURM jobs, each with a single Dask worker (``processes`` variable bellow),
and each worker uses 3 threads (``worker_process_threads`` variable bellow).
On this particular SLURM queue (sometimes called "partition" or "allocation") each node has 2GBi of RAM per core,
so we ask for 32GB of RAM and 16 cores per job.
So this configuration would use 60 SLURM jobs, use 180 Python threads, and 1920 GB of RAM in total.

.. code-block:: python

    class Job(dask_jobqueue.slurm.SLURMJob):
        # Rewrite the default, which is a property equal to cores/processes
        worker_process_threads = 3

    class Cluster(dask_jobqueue.SLURMCluster):
        job_cls = Job

    gb_per_job = 32
    jobs = 60
    processes = 1  # Single dask worker per slurm job
    gb_per_core = 2  # PSC "regular memory" nodes provide fixed 2GB / core
    cluster = Cluster(
        # Number of Dask workers per node
        processes=processes,
        # Regular memory node type on PSC bridges2
        queue="RM-shared",
        # dask_jobqueue requires cores and memory to be specified
        # We set them to match RM specs
        cores=gb_per_jon // gb_per_core,
        memory=f"{gb_per_job}GB",
        # Maximum walltime for the job, 6 hours.
        # SLURM will kill the job if it runs longer
        walltime="6:00:00",
    )

    # Run multiple jobs
    cluster.scale(jobs=jobs)

    # Alternatively to cluster.scale, can use adapt to run more jobs
    # cluster.adapt(maximum_jobs=100)

    client = dask.distributed.Client(cluster)


    # Your code, running catalog.compute() or catalog.to_hats()
    # df = catalog.compute()


    # Stop the cluster, it would ask SLURM to shut all the jobs down
    cluster.close()
    # Close the client
    client.close()

