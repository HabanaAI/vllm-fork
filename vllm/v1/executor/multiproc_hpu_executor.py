from vllm.v1.executor.multiproc_executor import MultiprocExecutor


class MultiprocHPUExecutor(MultiprocExecutor):

    def initialize(self, num_hpu_blocks: int) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        self.collective_rpc("initialize_cache", args=(num_hpu_blocks, ))