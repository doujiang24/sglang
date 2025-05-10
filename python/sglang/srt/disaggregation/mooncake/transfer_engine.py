import json
import math
import logging
import resource
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class MooncakeTransferEngine:

    def __init__(self, hostname: str, gpu_id: int, ib_device: Optional[str] = None):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run SGLang with MooncakeTransferEngine."
            ) from e

        self.engine = TransferEngine()
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.ib_device = ib_device
        self.raw_memory_segments = []

        self.initialize(
            hostname=self.hostname,
            device_name=self.ib_device,
        )
        self.session_id = f"{self.hostname}:{self.engine.get_rpc_port()}"

    def register(self, ptr, length):
        logger.info(f"ptr: {ptr}, length: {length}")
        self.raw_memory_segments.append((ptr, length))

    def register_submit(self):
        pagesize = resource.getpagesize()
        sorted_segments = sorted(self.raw_memory_segments, key= lambda x: x[0])
        merged_segments = []
        (cur_addr, cur_len) = sorted_segments[0]
        for (addr, len) in sorted_segments:
            cur_page_aligned = math.ceil((cur_addr + cur_len) / pagesize) * pagesize
            if addr <= cur_page_aligned:
                cur_len = addr + len - cur_addr
            else:
                merged_segments.append((cur_addr, cur_len))
                cur_addr, cur_len = addr, len
        merged_segments.append((cur_addr, cur_len))
        for (addr, len) in merged_segments:
            ptr = math.floor(addr / pagesize) * pagesize
            length = math.ceil((addr + len) / pagesize) * pagesize - ptr
            logger.info(f"register submmit, ptr: {ptr}, length: {length}")
            ret_value = self.engine.register_memory(ptr, length)
            if ret_value != 0:
                logger.error(f"Mooncake memory registration failed, ptr: {ptr}, length: {length}")
                raise RuntimeError("Mooncake memory registration failed.")

    def deregister(self, ptr):
        ret_value = self.engine.unregister_memory(ptr)
        if ret_value != 0:
            logger.error("Mooncake memory deregistration failed.")
            raise RuntimeError("Mooncake memory deregistration failed.")

    def initialize(
        self,
        hostname: str,
        device_name: Optional[str],
    ) -> None:
        """Initialize the mooncake instance."""
        ret_value = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            "rdma",
            device_name if device_name is not None else "",
        )
        if ret_value != 0:
            logger.error("Mooncake Transfer Engine initialization failed.")
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""

        ret = self.engine.transfer_sync_write(
            session_id, buffer, peer_buffer_address, length
        )
        if ret < 0:
            logger.error("Mooncake Transfer Engine Return Error.")
            raise RuntimeError("Mooncake Transfer Engine Return Error.")
        return ret

    def get_localhost(self):
        return self.hostname

    def get_session_id(self):
        return self.session_id
