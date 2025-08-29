======================================
Troubleshooting Frequent Dask Problems
======================================

-------------------------
Balancing Memory and Data
-------------------------

Most of these problems are a result of having too little memory or not
having it usefully apportioned.


Starting Small With Data
------------------------

It is very useful to limit the amount of data with up-front filters,
get a result, and then gradually widen the filters.

You can limit which columns are loaded with
``lsdb.open_catalog(columns=...)``, and you can limit the region of
the sky processed by
``lsdb.open_catalog(search_filter=lsdb.ConeSearch(...))`` (or
``lsdb.BoxSearch(...)``).  You can (and should!) combine these:

.. code-block:: python

    gaia_sm = lsdb.open_catalog(
        gaia_root,
        columns=['ra', 'dec', 'source_id'],
        search_filter=lsdb.ConeSearch(280, -60, radius_arcsec=1800)
        )

You can see the number of partitions remaining with the property
``.npartitions``, which is also displayed every time a catalog is
displayed as the output of a Jupyter cell.

Another way to reduce the load is to use the partition indexer after
the catalog is opened, with an expression like ``cat.partitions[0]``
to get a catalog with only a single partition.  This works for a range
of partitions, thus, ``cat.partitions[0:4]`` to get 4, and so on.


Working Your Way Upward
-----------------------

When you decide that you want to give each worker more memory, it's
often a good idea to reduce the number of workers by the same factor,
in order to keep your approximate memory footprint the same size.  If
this is not enough, for example:

.. code-block:: python

    from dask.distributed import Client
    client = Client(n_workers=16, memory_limit="8GB")

try this type of change:

.. code-block:: python

    from dask.distributed import Client
    client = Client(n_workers=8, memory_limit="16GB")


-----------------------------
Guide to common Dask messages
-----------------------------

Dask can produce a lot of messages during a given run, at a variety of
log levels.  It can be difficult at times to assess the true severity
and impact of these messages on the success of your LSDB job.  This
guide aims to help you do just that.

Perhaps you have a cluster running?
-----------------------------------

.. code-block:: output
    :class: no-copybutton

    lib/python3.12/site-packages/distributed/node.py:187: UserWarning: Port 8787 is already in use.
    Perhaps you already have a cluster running?
    Hosting the HTTP server on port 36727 instead
    warnings.warn(

This is a warning you will invariably see on a shared cluster.  Dask
creates a server to which you can connect if you want to observe the
Dask dashboard, which shows the status of all ongoing jobs.  If you're
running on your own computer, then port 8787 is likely to be free, and
so Dask warns you if it discovers that it isn't, because it probably
means that you left some Dask job running.

But on a cluster, it only means that you're using the machine with
many other people, so port 8787 is quite likely *not* to be free.
Dask then creates its dashboard server on a randomly chosen port and
puts that port number in the warning.  This isn't even a warning,
really.  It's like Standard Operating Procedure on a cluster.

Finding the wandering dashboard
...............................

On your local machine, you can view the dashboard by simply pointing
your browser to http://localhost:8787 .  On a cluster, it's likely
that this port isn't available to your local browser, due to firewall
rules.  As a result, you are likely obliged to create an ssh tunnel
that brings that port to your local machine so that you *can* visit
it, something like this (supposing that the port in the warning is
32120 and your cluster is ``monster.cluster.edu``):

.. code-block:: shell

    ssh $USER@monster.cluster.edu -L 32120:localhost:32120

after which you can visit http://localhost:32120 .

Note that this dashboard can be an elegant way of finding out how much
memory your workers were actually using, if you over-allocated, and
also a way of noticing how long it takes the cluster to set up versus
compute (more workers take longer), and so on.


Pausing workers
---------------

I didn't press pause!  Who did?

.. code-block:: output
    :class: no-copybutton

    2025-07-24 11:13:08,249 - distributed.worker.memory - WARNING - Worker is at 80% memory usage. Pausing worker.  Process memory: 1.49 GiB -- Worker memory limit: 1.86 GiB

This means that the ``memory_limit=`` argument you gave to your
``Client`` constructor turned out to be a bit lower than you needed.
Now one of your workers has almost used up their allocation and Dask,
rather than killing it, has decided to pause it to see whether the
worker might take a moment to do some garbage collection or otherwise
return some resources.  It's a real cross-fingers moment for Dask, but
it works sometimes.  It also allows the other workers to make
progress, if it turns out that one worker has put more on its plate
than it expected.

It's not a good sign.  It means that your processing is going to take
longer than you had hoped.  But it's not *fatal*.  Not yet.  It can be
a preamble to a dead job.


Stream closed terrors
---------------------

Dask clients can be used as context objects, and since the use of
context objects is a good practice in Python in general, you may be
tempted to compute your results like this:

.. code-block:: python

    with Client(n_workers=8, memory_limit="2GB") as client:
        results = unrealized.compute()

Often, this works fine, and you get a false sense of security and
superiority.  But nearly as often, this kind of thing fills up your
notebook at the end of the computation:

.. code-block:: output
    :class: no-copybutton

    2025-07-24 10:58:24,870 - distributed.worker - ERROR - Failed to communicate with scheduler during heartbeat.
    Traceback (most recent call last):
      File "site-packages/distributed/comm/tcp.py", line 226, in read
	frames_nosplit_nbytes_bin = await stream.read_bytes(fmt_size)
				    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    tornado.iostream.StreamClosedError: Stream is closed

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "site-packages/distributed/worker.py", line 1269, in heartbeat
	response = await retry_operation(
		   ^^^^^^^^^^^^^^^^^^^^^^
      File "site-packages/distributed/utils_comm.py", line 416, in retry_operation
	return await retry(
	       ^^^^^^^^^^^^
      File "site-packages/distributed/utils_comm.py", line 395, in retry
	return await coro()
	       ^^^^^^^^^^^^
      File "site-packages/distributed/core.py", line 1259, in send_recv_from_rpc
	return await send_recv(comm=comm, op=key, **kwargs)
	       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "site-packages/distributed/core.py", line 1018, in send_recv
	response = await comm.read(deserializers=deserializers)
		   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "site-packages/distributed/comm/tcp.py", line 237, in read
	convert_stream_closed_error(self, e)
      File "site-packages/distributed/comm/tcp.py", line 137, in convert_stream_closed_error
	raise CommClosedError(f"in {obj}: {exc}") from exc
    distributed.comm.core.CommClosedError: in <TCP (closed) ConnectionPool.heartbeat_worker local=tcp://127.0.0.1:50154 remote=tcp://127.0.0.1:36599>: Stream is closed


It turns out that these errors *don't matter*.  They have something to
do with the client being closed more quickly than it wants to be.
This is why so many of our demo notebooks create the client outside of
a context, and then ``client.close()`` at the end of the notebook.
The dark side of taking this approach is that if you forget to run
that cell, your Dask cluster will dangle, using up memory, until your
notebook kernel is restarted.


The general low memory error
----------------------------

And then there's this one, that doesn't seem to be related to the
amount of memory you gave your workers.  I mean, it says it isn't.
It's calling it "unmanaged" memory.

.. code-block:: output
    :class: no-copybutton

    2025-07-24 11:13:02,660 - distributed.worker.memory - WARNING - Unmanaged memory use is high. This may indicate a memory leak or the memory may not be released to the OS; see https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os for more information. -- Unmanaged memory: 1.31 GiB -- Worker memory limit: 1.86 GiB

And yet: giving your workers more memory often clears this up.

If it doesn't, the problem could be the task graph.  Perhaps try to
express your computation more idempotently, or produce intermediate
results.


The poison pill
---------------

Saving the worst for last.  This one is a simple warning that sounds
like the cluster has run into a minor problem that it's going to route
around with a little rescheduling.

.. code-block:: output
    :class: no-copybutton

    2025-07-24 11:32:00,670 - distributed.client - WARNING - Couldn't gather 1 keys, rescheduling (('repartitiontofewer-77ee1928ccf3f483f566fd6c17ee139b', 0),)

Nope.  This means you're **done**.  Your task will not complete.  If
you're watching your Dask dashboard at this point, you will see
that it seems to have frozen without explaining why.

Solution: you **must** find a way to give each worker more memory
until that warning goes away.  It's a low-memory problem.  It doesn't
say that.  The dashboard probably didn't even show workers running out
of memory.  But they did.  And it's even worse.  You'd better just
restart your kernel because you won't be able to close that old
client.  Tear it all down and start fresh.  Really fresh.


-----------------
Observed problems
-----------------

Problems that may not be accompanied by immediate error messages.


All workers are being killed in the beginning
---------------------------------------------

If you see that the pipeline failed fast after it started, it may be
due to a bug in the code, data access issues, or memory overflow.  For
the first two cases, you would see the appropriate error messages in
the logs.  If the message doesn't contain enough useful information,
you can try to run the pipeline with no ``Client`` object being
created.  In this case, Dask will use the default scheduler, which
will run tasks on the same Python process and give you a usual Python
traceback on the failure.

In the case of the memory overflow, Dask Dashboard will show red bars
in the memory usage chart, and logs will show messages like the
following:

.. code-block:: output
    :class: no-copybutton

    distributed.nanny.memory - WARNING - Worker tcp://127.0.0.1:49477 (pid=59029) exceeded 95% memory budget. Restarting...
    distributed.nanny - WARNING - Restarting worker
    KilledWorker: Attempted to run task ('read_pixel-_to_string_dtype-nestedframe-0c9d20582a6d2703d02a4835dddb05d2', 30904) on 4 different workers, but all those workers died while running it. The last worker that attempt to run the task was tcp://127.0.0.1:50761. Inspecting worker logs is often a good next step to diagnose what went wrong. For more information see https://distributed.dask.org/en/stable/killed.html.


All workers are being killed in the middle/end
----------------------------------------------

Some workflows can have a very unbalanced memory load, so just one or
few tasks would use much more memory than others.  You can diagnose
this by looking at the memory usage chart in Dask Dashboard, it would
show that only one worker is using much more memory than others.  In
such cases you may set the total memory limit ``memory_limit *
n_workers`` larger than the actual amount of memory on your system.
For example, if you have 16GB of RAM and you see that almost all of
the tasks need 1GB, while a single task needs 8GB, you can start a
cluster with this command:

.. code-block:: python

    from dask.distributed import Client
    client = Client(n_workers=8, memory_limit='8GB', threads_per_worker=1)


This approach can also help to speed up the computations, because it enables running with more workers.


I run ``.compute()``, but the Dask Dashboard is empty for a long time
---------------------------------------------------------------------

For large tasks, such as cross-matching or joining multiple
dozen-terabyte scale catalogs, Dask may spend a lot of time and memory
of the main process before any computation starts.  This happens
because Dask builds and optimizes the computation graph, which happens
on the main process (one you create ``Client`` on).
