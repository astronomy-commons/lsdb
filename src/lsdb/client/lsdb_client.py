from dask.distributed import Client
import ray
from ray.util.dask import enable_dask_on_ray, disable_dask_on_ray

class lsdb_client():
    def __init__(self, dask_on_ray=True, num_workers=4):
        self.dask_on_ray = dask_on_ray
        self.num_workers=num_workers
        self.start()

    def start(self):

        if self.dask_on_ray:
            self.client = ray.init(
                num_cpus=self.num_workers,
                _temp_dir="/data3/epyc/projects3/sam_hipscat"
            )
            enable_dask_on_ray()
        else:
            self.client = Client(n_workers=self.num_workers, threads_per_worker=1)

    def shutdown(self):
        if self.dask_on_ray:
            disable_dask_on_ray()
            ray.shutdown()
        else:
            self.client.close()