import torch
import numpy as np
from functools import cached_property
from arsenal import colors
from .lm_state import StatefulTokenizedLM
from ..util import escape, LazyByteProbs
from ..trie import TrieMode

# EOS byte constant - using 257 as the virtual EOS byte
EOS = 257


class LazyTrieState:
    """A lazy-evaluated state of a TokenByteTrie traversal.

    This class maintains the state of a language model while traversing a trie structure,
    lazily evaluating probabilities and maintaining the weight of the current path through the trie
    for beam search.

    Args:
        lm_state (StatefulTokenizedLM): Current language model state
        trie (TokenByteTrie): Trie structure mapping tokens to byte sequences
        node (int): Current node in the trie
        weight (float): Cumulative log probability of the path to this node
        mass (numpy.ndarray, optional): Masses for each node in the trie for the current state
        mode (TrieMode): Trie mode to use
        terminated (bool): Whether the state is terminated (EOS has been consumed)
    """

    def __init__(
        self,
        lm_state,
        trie,
        node,
        weight,
        mass=None,
        mode=TrieMode.WITH_EOS,
        terminated=False,
    ):
        self.lm_state = lm_state
        self.trie = trie
        self.node = node
        self.weight = weight
        self._mass = mass
        self._extend = None
        self.mode = mode
        self.root = self.trie.trie.root
        self.children = self.trie.trie.children
        self.terminated = terminated

    @classmethod
    def initial(cls, lm, trie, mode=TrieMode.WITH_EOS, initial_context=None):
        """Creates an initial trie state.

        Args:
            lm (genlm.backend.AsyncLM): Language model to use
            trie (TokenByteTrie): TokenByteTrie structure for byte-to-token mapping
            mode (TrieMode): Trie mode to use
            initial_context (list, optional): Initial context of token IDs for the LM.

        Returns:
            (LazyTrieState): Initial state at root of trie with weight 0.0
        """
        return cls(
            trie=trie,
            node=trie.trie.root,
            lm_state=StatefulTokenizedLM.initial(lm, initial_context=initial_context),
            weight=0.0,
            mode=mode,
        )

    @property
    def partial(self):
        """Returns the byte sequence corresponding to the current node in the trie."""
        return self.trie.trie.node2prefix[self.node]

    @property
    def mass(self):
        """Returns the log mass for each node in the trie.

        The mass at a node corresponds to the sum of the probabilities of all
        tokens which share the prefix (`self.partial`) represented by that node.

        Raises:
            ValueError: If state hasn't been materialized yet
        """
        if self._mass is None:
            raise ValueError("State is not yet materialized.")
        return self._mass

    def with_mode(self, mode):
        """Returns a new state with the given mode."""
        return LazyTrieState(
            lm_state=self.lm_state,
            trie=self.trie,
            node=self.node,
            weight=self.weight,
            mass=self._mass,
            mode=mode,
            terminated=self.terminated,
        )

    def actions(self):
        """Returns possible byte transitions from current node."""
        return self.children[self.node]

    def get_EOT(self):
        """Returns the end-of-token node if available from current position in the trie."""
        return self.children[self.node].get(self.trie.trie.eot_token)

    def __lshift__(self, b):
        """Transitions to a new state by consuming a byte.

        Args:
            b (int): Byte to consume (0-255 for regular bytes, 257 for EOS, 258+ for special tokens)

        Returns:
            (LazyTrieState|None): New state after consuming byte, or None if transition invalid (terminated or EOS)
        """
        if self.terminated:
            return None

        if node := self.children[self.node].get(b):
            mass = self.mass

            # Special token virtual bytes: commit to LM state and return to root
            if b in self.trie.trie.special_token_bytes:
                token_id = self.trie.trie.special_token_bytes[b]
                return LazyTrieState(
                    lm_state=self.lm_state << token_id,
                    trie=self.trie,
                    mass=None,  # needs rematerialization
                    node=self.root,
                    weight=self.weight + mass[node] - mass[self.node],
                    mode=self.mode,
                    terminated=False,
                )

            return LazyTrieState(
                lm_state=self.lm_state,
                trie=self.trie,
                mass=mass,
                node=node,
                weight=self.weight + mass[node] - mass[self.node],
                mode=self.mode,
                terminated=b == EOS,
            )

    def extend(self):
        """Extends current state by consuming an end-of-token if possible.

        Returns:
            (LazyTrieState|None): New state after consuming EOT, or None if not possible
        """
        if self._extend is None:
            if (eot_node := self.get_EOT()) is not None:
                mass = self.mass
                self._extend = LazyTrieState(
                    lm_state=self.lm_state
                    << int(self.trie.trie.leaf2token_id[eot_node]),
                    trie=self.trie,
                    node=self.root,
                    weight=self.weight + mass[eot_node] - mass[self.node],
                    mode=self.mode,
                )
        return self._extend

    @cached_property
    def logp_next(self):
        """Computes log probabilities for next possible transitions.

        Returns:
            (LazyByteProbs): Lazy log probability distribution over possible next bytes
        """
        n_special = len(self.trie.trie.special_tokens)
        logps = np.full(258 + n_special, -np.inf)
        mass = self.mass
        logZ = mass[self.node]

        for byte, node in self.actions().items():
            logps[byte if byte is not None else 256] = mass[node] - logZ

        special_token_names = [
            repr(t) for t in self.trie.trie.special_tokens
        ]
        return LazyByteProbs(logps, special_token_names=special_token_names)

    async def materialize(self):
        """Materializes the masses for each node in the trie for the current state.

        This makes a call to the language model and the underlying trie.

        Returns:
            (LazyTrieState): Self with materialized masses
        """
        if self._mass is None:
            logp_next = await self.lm_state.logp_next()
            vocab_size = len(self.trie.trie.decode)
            if logp_next.shape[0] > vocab_size:
                logp_next = logp_next[:vocab_size]
                logp_next = logp_next - torch.logsumexp(logp_next, dim=0)
            log_mass = await self.trie.weight_sum(torch.exp(logp_next), self.mode)
            mass = torch.log(log_mass)
            self._mass = mass.cpu().numpy()
        return self

    def __repr__(self):
        context = colors.green % ("|" + escape(bytes(self.partial)))
        if self.terminated:
            context += colors.yellow % "<EOS>"
        return f"{self.weight:.2f}: {self.lm_state}" + context

    async def cleanup(self):
        """Cleans up resources used by the trie."""
        await self.trie.cleanup()
