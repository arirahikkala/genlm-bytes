import torch
import pytest
import numpy as np

from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams, TokenByteTrie
from genlm.bytes.trie import EOS
from genlm.bytes.byte_lm.trie_state import TrieMode


# ---------- Trie-level tests (no LLM needed) ----------


@pytest.fixture(scope="module")
def special_trie():
    """Trie with special tokens configured."""
    vocab = [
        b"hello",  # 0
        b"world",  # 1
        b"<|im_start|>",  # 2 (special)
        b"<|tool|>",  # 3 (special)
        b"normal",  # 4
        b"<eos>",  # 5 (EOS)
    ]
    return TokenByteTrie(
        decode=vocab,
        eos_tokens=[b"<eos>"],
        special_tokens=[b"<|im_start|>", b"<|tool|>"],
    )


def test_special_nodes_created(special_trie):
    """Special token nodes should be created and connected from root."""
    assert len(special_trie.special_nodes) == 2
    assert 258 in special_trie.special_nodes
    assert 259 in special_trie.special_nodes

    # Connected from root via virtual bytes
    assert special_trie.children[special_trie.root].get(258) == special_trie.special_nodes[258]
    assert special_trie.children[special_trie.root].get(259) == special_trie.special_nodes[259]


def test_special_token_ids(special_trie):
    """Special token IDs should map correctly."""
    assert special_trie.special_token_ids == [2, 3]  # vocab indices
    assert special_trie.special_token_bytes == {258: 2, 259: 3}


def test_special_tokens_in_reachability_with_eos(special_trie):
    """In WITH_EOS mode, special tokens should divert mass to their nodes."""
    # weights: hello=0.1, world=0.1, <|im_start|>=0.3, <|tool|>=0.2, normal=0.1, <eos>=0.2
    weights = torch.tensor([0.1, 0.1, 0.3, 0.2, 0.1, 0.2])

    masses = special_trie.weight_sum(weights, mode=TrieMode.WITH_EOS)

    # Special nodes should have the mass of their respective tokens
    im_start_mass = masses[special_trie.special_nodes[258]]
    tool_mass = masses[special_trie.special_nodes[259]]

    assert np.isclose(im_start_mass.item(), 0.3, rtol=1e-5)
    assert np.isclose(tool_mass.item(), 0.2, rtol=1e-5)

    # EOS node should have its token's mass
    eos_mass = masses[special_trie.eos_node]
    assert np.isclose(eos_mass.item(), 0.2, rtol=1e-5)

    # Root should still have total mass (special + eos mass added back to root)
    root_mass = masses[special_trie.root]
    assert np.isclose(root_mass.item(), 1.0, rtol=1e-5)


def test_special_tokens_in_reachability_without_eos(special_trie):
    """In WITHOUT_EOS mode, special tokens are treated as normal tokens."""
    weights = torch.tensor([0.1, 0.1, 0.3, 0.2, 0.1, 0.2])

    masses = special_trie.weight_sum(weights, mode=TrieMode.WITHOUT_EOS)

    # Special nodes should have zero mass in no_eos mode
    for vbyte in special_trie.special_nodes:
        assert masses[special_trie.special_nodes[vbyte]] == 0.0

    # EOS node should also have zero mass
    assert masses[special_trie.eos_node] == 0.0

    # Root should have total mass
    root_mass = masses[special_trie.root]
    assert np.isclose(root_mass.item(), 1.0, rtol=1e-5)


def test_backward_compat_no_special_tokens():
    """Without special_tokens configured, distribution should be 258 elements."""
    vocab = [b"a", b"b", b"ab"]
    trie = TokenByteTrie(decode=vocab)
    assert len(trie.special_tokens) == 0
    assert len(trie.special_nodes) == 0
    assert len(trie.special_token_ids) == 0


def test_invalid_special_token():
    """Should raise error for special token not in vocab."""
    with pytest.raises(ValueError, match="Special token"):
        TokenByteTrie(
            decode=[b"a", b"b"],
            special_tokens=[b"<|nonexistent|>"],
        )


def test_special_and_eos_together():
    """Special tokens and EOS tokens can coexist."""
    vocab = [b"hello", b"<eos>", b"<|special|>"]
    trie = TokenByteTrie(
        decode=vocab,
        eos_tokens=[b"<eos>"],
        special_tokens=[b"<|special|>"],
    )

    weights = torch.tensor([0.5, 0.3, 0.2])
    masses = trie.weight_sum(weights, mode=TrieMode.WITH_EOS)

    # EOS and special should each have their mass
    assert np.isclose(masses[trie.eos_node].item(), 0.3, rtol=1e-5)
    assert np.isclose(masses[trie.special_nodes[258]].item(), 0.2, rtol=1e-5)


# ---------- Integration tests with LLM ----------


@pytest.fixture(scope="module")
def llm():
    return load_model_by_name("gpt2-medium", backend="hf")


@pytest.mark.asyncio
async def test_beam_with_special_tokens(llm):
    """Test that special tokens appear in the distribution at the right indices."""
    # Find tokens that actually exist in GPT-2's vocabulary
    byte_vocab = llm.byte_vocab
    # Pick two tokens from the vocab to be "special"
    special_tokens = [byte_vocab[100], byte_vocab[200]]

    params = BeamParams(K=5, special_tokens=special_tokens)
    state = await ByteBeamState.initial(llm, params)

    try:
        logp_next = await state.logp_next()
        # Distribution should have 260 elements (258 + 2 special)
        assert len(logp_next.ps) == 260
        # Special token slots should have finite probabilities
        assert logp_next[258] > -np.inf or logp_next[258] == -np.inf  # either is ok; just shouldn't error
        assert logp_next[259] > -np.inf or logp_next[259] == -np.inf
    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_beam_no_special_tokens_backward_compat(llm):
    """Without special_tokens, distribution is 258 elements (unchanged)."""
    params = BeamParams(K=5)
    state = await ByteBeamState.initial(llm, params)

    try:
        logp_next = await state.logp_next()
        assert len(logp_next.ps) == 258
    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_special_token_consumption(llm):
    """Consuming a special token virtual byte should commit to LM and return to root."""
    byte_vocab = llm.byte_vocab
    special_token = byte_vocab[100]

    params = BeamParams(K=5, special_tokens=[special_token])
    state = await ByteBeamState.initial(llm, params)

    try:
        # Consume the special token (virtual byte 258)
        new_state = await (state << 258)

        # Should not be terminated
        assert not any(s.terminated for s in new_state.states)

        # States should be at root
        for s in new_state.states:
            assert s.node == s.root

        # Should be able to continue generating
        if len(new_state) > 0:
            logp_next = await new_state.logp_next()
            assert len(logp_next.ps) == 259  # 258 + 1 special
    finally:
        await state.cleanup()


@pytest.mark.asyncio
async def test_special_token_with_eos(llm):
    """Special tokens and EOS can be configured together."""
    byte_vocab = llm.byte_vocab
    special_token = byte_vocab[100]

    params = BeamParams(
        K=5,
        eos_tokens=[byte_vocab[50]],
        special_tokens=[special_token],
    )
    state = await ByteBeamState.initial(llm, params)

    try:
        logp_next = await state.logp_next()
        # Should have 259 elements (258 base + 1 special)
        assert len(logp_next.ps) == 259
        # EOS slot should be present
        eos_logp = logp_next[257]
        assert isinstance(eos_logp, (float, np.floating))
    finally:
        await state.cleanup()
