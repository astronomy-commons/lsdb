Guide to Dask messages (troubleshooting)
========================================

Dask can produce a lot of messages during a given run, at a variety of
log levels.  It can be difficult at times to asses the true severity
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

If it doesn't, the problem could be the task graph.  Any way you can
express your computation more idempotently?  I mean, maybe not, but
try.  Or have more intermediate results.


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
