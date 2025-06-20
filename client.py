import random
from typing import List
from labrpc.labrpc import ClientEnd, TimeoutError
from server import GetArgs, GetReply, PutAppendArgs, PutAppendReply


def nrand() -> int:
    """Return a fresh 62-bit random integer."""
    return random.getrandbits(62)


class Clerk:
    """
    One Clerk per application thread (tests guarantee at most one outstanding
    RPC per Clerk).  The Clerk is *not* thread-safe.
    """
    def __init__(self, servers: List[ClientEnd], cfg):
        self.servers = servers
        self.cfg = cfg                  # test-supplied config object
        self.client_id = nrand()        # unique across the whole test run
        self.seq = 0                    # monotonically increasing RPC number
        self.nservers = len(servers)
        self.nreplicas = getattr(cfg, "nreplicas", 1)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _shard_for_key(self, key: str) -> int:
        """Deterministic shard assignment: int(key) % N."""
        try:
            k = int(key)
        except ValueError:
            # Fallback for non-numeric keys (rare in the tests)
            k = sum(ord(c) for c in key)
        return k % self.nservers

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get(self, key: str) -> str:
        """Linearizable Get with infinite retry in the face of failures."""
        args = GetArgs(key=key,
                       client_id=self.client_id,
                       seq=self._next_seq())
        shard = self._shard_for_key(key)

        while True:
            for r in range(self.nreplicas):
                idx = (shard + r) % self.nservers
                try:
                    reply: GetReply = self.servers[idx].call("KVServer.Get", args)
                    return reply.value
                except TimeoutError:
                    # RPC lost – try next replica
                    continue

    # shared implementation for Put and Append
    def _put_append(self, key: str, value: str, op: str) -> str:
        args = PutAppendArgs(key=key,
                             value=value,
                             op=op,
                             client_id=self.client_id,
                             seq=self._next_seq())
        shard = self._shard_for_key(key)

        while True:
            for r in range(self.nreplicas):
                idx = (shard + r) % self.nservers
                try:
                    reply: PutAppendReply = self.servers[idx].call(f"KVServer.{op}", args)
                    return reply.value          # Put ⇒ “”, Append ⇒ previous value
                except TimeoutError:
                    continue

    # ---------------------------------------------------------------------
    # Convenience wrappers
    # ---------------------------------------------------------------------
    def put(self, key: str, value: str) -> None:
        self._put_append(key, value, "Put")

    def append(self, key: str, value: str) -> str:
        return self._put_append(key, value, "Append")
