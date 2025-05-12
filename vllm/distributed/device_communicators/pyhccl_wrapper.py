import ctypes
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

from vllm.logger import init_logger

logger = init_logger(__name__)


hcclResult_t = ctypes.c_int
hcclComm_t = ctypes.c_void_p


class hcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_uint8 * 1024), ("length", ctypes.c_size_t)]


hpuStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

hcclDataType_t = ctypes.c_int


class hcclDataTypeEnum:
    hcclInt8 = 0
    hcclChar = 0
    hcclUint8 = 1
    hcclInt32 = 2
    hcclInt = 2
    hcclUint32 = 3
    hcclInt64 = 4
    hcclUint64 = 5
    hcclFloat16 = 6
    hcclHalf = 6
    hcclFloat32 = 7
    hcclFloat = 7
    hcclFloat64 = 8
    hcclDouble = 8
    hcclBfloat16 = 9
    hcclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.hcclInt8
        if dtype == torch.uint8:
            return cls.hcclUint8
        if dtype == torch.int32:
            return cls.hcclInt32
        if dtype == torch.int64:
            return cls.hcclInt64
        if dtype == torch.float16:
            return cls.hcclFloat16
        if dtype == torch.float32:
            return cls.hcclFloat32
        if dtype == torch.float64:
            return cls.hcclFloat64
        if dtype == torch.bfloat16:
            return cls.hcclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


hcclRedOp_t = ctypes.c_int


class hcclRedOpTypeEnum:
    hcclSum = 0
    hcclProd = 1
    hcclMin = 2
    hcclMax = 3
    hcclAvg = 4
    hcclOpNone = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.hcclSum
        if op == ReduceOp.PRODUCT:
            return cls.hcclProd
        if op == ReduceOp.MAX:
            return cls.hcclMax
        if op == ReduceOp.MIN:
            return cls.hcclMin
        if op == ReduceOp.AVG:
            return cls.hcclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class HCCLLibrary:
    exported_functions = [
        # synStatus SYN_API_CALL synStreamCreateGeneric(synStreamHandle*  pStreamHandle,
        #                                       const synDeviceId deviceId,
        #                                       const uint32_t    flags);
        Function(
            "synStreamCreateGeneric",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32],
        ),
        # synStatus SYN_API_CALL synStreamSynchronize( const synStreamHandle streamHandle );
        Function(
            "synStreamSynchronize",
            ctypes.c_int32,
            [ctypes.c_void_p],
        ),
        # const char* hcclGetErrorString(hcclResult_t result);
        Function("hcclGetErrorString", ctypes.c_char_p, [hcclResult_t]),
        # hcclResult_t hcclGetVersion(int* version);
        Function("hcclGetVersion", hcclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        # hcclResult_t hcclGetUniqueId(hcclUniqueId* uniqueId);
        Function("hcclGetUniqueId", hcclResult_t, [ctypes.POINTER(hcclUniqueId)]),
        # hcclResult_t hcclCommInitRank(hcclComm_t* comm, int nranks, hcclUniqueId commId, int rank);
        # note that hcclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function(
            "hcclCommInitRank",
            hcclResult_t,
            [ctypes.POINTER(hcclComm_t), ctypes.c_int, hcclUniqueId, ctypes.c_int],
        ),
        # hcclResult_t hcclAllReduce(const void*    sendbuff,
        #                    void*          recvbuff,
        #                    size_t         count,
        #                    hcclDataType_t datatype,
        #                    hcclRedOp_t    reduceOp,
        #                    hcclComm_t     comm,
        #                    void*          stream_handle);
        # note that ctypes.c_void_p is a pointer type, so the last argument
        # is a pointer
        Function(
            "hcclAllReduce",
            hcclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                hcclDataType_t,
                hcclRedOp_t,
                hcclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # hcclResult_t hcclAllGather(const void*    sendbuff,
        #                    void*          recvbuff,
        #                    size_t         sendcount,
        #                    hcclDataType_t datatype,
        #                    hcclComm_t     comm,
        #                    void*          stream_handle);
        # note that ctypes.c_void_p is a pointer type, so the last argument
        # is a pointer
        Function(
            "hcclAllGather",
            hcclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                hcclDataType_t,
                hcclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # hcclResult_t hcclReduceScatter(const void*    sendbuff,
        #                        void*          recvbuff,
        #                        size_t         recvcount,
        #                        hcclDataType_t datatype,
        #                        hcclRedOp_t    reduceOp,
        #                        hcclComm_t     comm,
        #                        void*          stream_handle);
        # note that ctypes.c_void_p is a pointer type, so the last argument
        # is a pointer
        Function(
            "hcclReduceScatter",
            hcclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                hcclDataType_t,
                hcclRedOp_t,
                hcclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # hcclResult_t hcclSend(const void* sendbuff,
        #                       size_t count,
        #                       hcclDataType_t datatype,
        #                       int peer,
        #                       hcclComm_t comm,
        #                       void* stream);
        Function(
            "hcclSend",
            hcclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_size_t,
                hcclDataType_t,
                ctypes.c_int,
                hcclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # hcclResult_t hcclRecv(void* recvbuff,
        #                       size_t count,
        #                       hcclDataType_t datatype,
        #                       int peer,
        #                       hcclComm_t comm,
        #                       void* stream);
        Function(
            "hcclRecv",
            hcclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_size_t,
                hcclDataType_t,
                ctypes.c_int,
                hcclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # hcclResult_t hcclBroadcast(const void*    sendbuff,
        #                    void*          recvbuff,
        #                    size_t         count,
        #                    hcclDataType_t datatype,
        #                    int            root,
        #                    hcclComm_t     comm,
        #                    void*          stream_handle);
        Function(
            "hcclBroadcast",
            hcclResult_t,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                hcclDataType_t,
                ctypes.c_int,
                hcclComm_t,
                ctypes.c_void_p,
            ],
        ),
        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # hcclResult_t hcclCommDestroy(hcclComm_t comm);
        Function("hcclCommDestroy", hcclResult_t, [hcclComm_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):

        self.streamHandle = ctypes.c_void_p()

        so_file = so_file or "libSynapse.so"

        try:
            if so_file not in HCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                HCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = HCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load HCCL library from %s ."
                "It is expected if you are not running on Gaudi."
                "Otherwise, the hccl library might not exist, be corrupted "
                "or it does not support the current platform %s.", so_file,
                platform.platform())
            raise e

        if so_file not in HCCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in HCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            HCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = HCCLLibrary.path_to_dict_mapping[so_file]
    
    def synStreamCreateGeneric(self):
        res = self._funcs["synStreamCreateGeneric"](ctypes.byref(self.streamHandle), 0, 0)
        assert res == 0

    def synStreamSynchronize(self):
        res = self._funcs["synStreamSynchronize"](self.streamHandle)
        assert res == 0

    def hcclGetErrorString(self, result: hcclResult_t) -> str:
        return self._funcs["hcclGetErrorString"](result).decode("utf-8")

    def HCCL_CHECK(self, result: hcclResult_t) -> None:
        if result != 0:
            error_str = self.hcclGetErrorString(result)
            raise RuntimeError(f"HCCL error: {error_str}")

    def hcclGetVersion(self) -> str:
        version = ctypes.c_int()
        self.HCCL_CHECK(self._funcs["hcclGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        # something like 2604 --> "2.6.4"
        major = version_str[0].lstrip("0")
        minor = version_str[1].lstrip("0")
        patch = version_str[2:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def hcclGetUniqueId(self) -> hcclUniqueId:
        unique_id = hcclUniqueId()
        self.HCCL_CHECK(self._funcs["hcclGetUniqueId"](
            ctypes.byref(unique_id)))
        return unique_id

    def hcclCommInitRank(self, world_size: int, unique_id: hcclUniqueId,
                         rank: int) -> hcclComm_t:
        comm = hcclComm_t()
        self.HCCL_CHECK(self._funcs["hcclCommInitRank"](ctypes.byref(comm),
                                                        world_size, unique_id,
                                                        rank))
        return comm

    def hcclAllReduce(self, sendbuff: ctypes.c_void_p, recvbuff: ctypes.c_void_p,
                      count: int, datatype: int, op: int, comm: hcclComm_t,
                      stream: ctypes.c_void_p) -> None:
        # `datatype` actually should be `hcclDataType_t`
        # and `op` should be `hcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.HCCL_CHECK(self._funcs["hcclAllReduce"](sendbuff, recvbuff, count,
                                                     datatype, op, comm,
                                                     stream))

    def hcclReduceScatter(self, sendbuff: ctypes.c_void_p, recvbuff: ctypes.c_void_p,
                          count: int, datatype: int, op: int, comm: hcclComm_t,
                          stream: ctypes.c_void_p) -> None:
        # `datatype` actually should be `hcclDataType_t`
        # and `op` should be `hcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.HCCL_CHECK(self._funcs["hcclReduceScatter"](sendbuff, recvbuff,
                                                         count, datatype, op,
                                                         comm, stream))

    def hcclAllGather(self, sendbuff: ctypes.c_void_p, recvbuff: ctypes.c_void_p,
                      count: int, datatype: int, comm: hcclComm_t,
                      stream: ctypes.c_void_p) -> None:
        # `datatype` actually should be `hcclDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.HCCL_CHECK(self._funcs["hcclAllGather"](sendbuff, recvbuff, count,
                                                     datatype, comm, stream))

    def hcclSend(self, sendbuff: ctypes.c_void_p, count: int, datatype: int,
                 dest: int, comm: hcclComm_t, stream: ctypes.c_void_p) -> None:
        self.HCCL_CHECK(self._funcs["hcclSend"](sendbuff, count, datatype,
                                                dest, comm, stream))

    def hcclRecv(self, recvbuff: ctypes.c_void_p, count: int, datatype: int,
                 src: int, comm: hcclComm_t, stream: ctypes.c_void_p) -> None:
        self.HCCL_CHECK(self._funcs["hcclRecv"](recvbuff, count, datatype, src,
                                                comm, stream))

    def hcclBroadcast(self, sendbuff: ctypes.c_void_p, recvbuff: ctypes.c_void_p,
                      count: int, datatype: int, root: int, comm: hcclComm_t,
                      stream: ctypes.c_void_p) -> None:
        self.HCCL_CHECK(self._funcs["hcclBroadcast"](sendbuff, recvbuff, count,
                                                     datatype, root, comm,
                                                     stream))

    def hcclCommDestroy(self, comm: hcclComm_t) -> None:
        self.HCCL_CHECK(self._funcs["hcclCommDestroy"](comm))


__all__ = [
    "HCCLLibrary", "hcclDataTypeEnum", "hcclRedOpTypeEnum", "hcclUniqueId",
    "hcclComm_t", "ctypes.c_void_p", "ctypes.c_void_p"
]
