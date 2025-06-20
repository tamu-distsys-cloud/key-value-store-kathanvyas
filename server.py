# Source code was built using generative application like ChapGPT O3
import logging
import threading
from typing import Dict

debugging = False
def debug(fmt, *args):           # optional printf-style server tracing
    if debugging:
        logging.info(fmt % args)

# -------------------------------------------------------------------------
# RPC argument & reply structures  (pickle-serialised by labrpc/labgob)
# -------------------------------------------------------------------------
class PutAppendArgs:
    def __init__(self, key: str, value: str, op: str,
                 client_id: int, seq: int):
        self.key = key
        self.value = value
        self.op = op              # "Put" or "Append"
        self.client_id = client_id
        self.seq = seq            # monotonically increasing per client

class PutAppendReply:
    def __init__(self, value: str = ""):
        # For Append we return the *previous* value; for Put this is “”.
        self.value = value

class GetArgs:
    def __init__(self, key: str, client_id: int = 0, seq: int = 0):
        self.key = key
        self.client_id = client_id
        self.seq = seq

class GetReply:
    def __init__(self, value: str = ""):
        self.value = value

# -------------------------------------------------------------------------
# Key-Value server
# -------------------------------------------------------------------------
class KVServer:
    def __init__(self, cfg):
        self.cfg = cfg                    # cluster-wide test harness object
        self.mu = threading.Lock()

        # persistent state
        self.store: Dict[str, str] = {}   # key → current value
        # at-most-once bookkeeping:  client_id → (latest_seq, last_reply)
        self.dup: Dict[int, tuple] = {}

        # filled lazily – this server’s index in cfg.kvservers[]
        self.me = None

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _ensure_me(self):
        if self.me is None:
            # O(N) once per process – fine for the small lab clusters
            self.me = self.cfg.kvservers.index(self)

    def _primary_shard(self, key: str) -> int:
        try:
            k = int(key)
        except ValueError:
            k = sum(ord(c) for c in key)
        return k % self.cfg.nservers

    def _owns_key(self, key: str) -> bool:
        """
        A server owns a key if it is the primary shard or one of the
        (nreplicas-1) followers immediately following the primary.
        """
        self._ensure_me()
        nrep = getattr(self.cfg, "nreplicas", 1)
        dist = (self.me - self._primary_shard(key)) % self.cfg.nservers
        return dist < nrep          # dist==0 ⇒ primary; 0<dist<nrep ⇒ replica

    # best-effort synchronous replication to followers in the same process
    def _replicate(self, key: str, new_val: str):
        nrep = getattr(self.cfg, "nreplicas", 1)
        if nrep <= 1:
            return
        primary = self._primary_shard(key)
        for r in range(1, nrep):
            sid = (primary + r) % self.cfg.nservers
            if sid == self.me:
                continue
            follower = self.cfg.kvservers[sid]
            follower._apply_update(key, new_val)

    def _apply_update(self, key: str, new_val: str):
        with self.mu:
            self.store[key] = new_val

    # ---------------------------------------------------------------------
    # RPC handlers
    # ---------------------------------------------------------------------
    def Get(self, args: GetArgs) -> GetReply:
        if not self._owns_key(args.key):
            # Caller hit the wrong shard – empty string signals rejection
            return GetReply("")
        with self.mu:
            val = self.store.get(args.key, "")
            # (Duplicate detection for Get isn’t strictly required,
            #  but keeping the table symmetrical is harmless.)
            self.dup[args.client_id] = (args.seq, val)
            return GetReply(val)

    def Put(self, args: PutAppendArgs) -> PutAppendReply:
        return self._put_append(args, is_append=False)

    def Append(self, args: PutAppendArgs) -> PutAppendReply:
        return self._put_append(args, is_append=True)

    # shared Put/Append implementation
    def _put_append(self, args: PutAppendArgs, *, is_append: bool) -> PutAppendReply:
        if not self._owns_key(args.key):
            return PutAppendReply("")          # reject – wrong shard

        with self.mu:
            last_seq, last_reply = self.dup.get(args.client_id, (-1, ""))
            if args.seq <= last_seq:
                # duplicate RPC – just return the cached reply
                return PutAppendReply(last_reply)

            prev_val = self.store.get(args.key, "")
            new_val = args.value if not is_append else prev_val + args.value
            self.store[key := args.key] = new_val

            # remember for at-most-once
            reply_val = "" if not is_append else prev_val
            self.dup[args.client_id] = (args.seq, reply_val)

        # replicate outside the critical section (best effort)
        self._replicate(key, new_val)
        return PutAppendReply(reply_val)
