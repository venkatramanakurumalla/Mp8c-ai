import torch
import random
import cmath  # for complex in FFT

class PureHDCTextGen:
    def __init__(
        self,
        vocab,
        dim=4096,
        seed=42,
        beam_width=5,
        num_noisy_copies=7,
        max_positions=512,
        noise_level=0.08
    ):
        torch.manual_seed(seed)
        random.seed(seed)

        self.dim = dim
        self.vocab = list(dict.fromkeys(vocab))
        self.beam_width = beam_width
        self.num_noisy_copies = num_noisy_copies
        self.noise_level = noise_level

        # Item memory: bipolar unit vectors
        self.item_memory = {}
        for word in self.vocab:
            v = torch.randn(dim).sign().float()
            v /= torch.norm(v) + 1e-8
            self.item_memory[word] = v

        # Positional memory: unique bipolar vectors
        self.pos_memory = torch.randn(max_positions, dim).sign().float()
        self.pos_memory /= torch.norm(self.pos_memory, dim=1, keepdim=True) + 1e-8

    def circular_convolution(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """HRR-style binding via FFT circular convolution"""
        fa = torch.fft.fft(a)
        fb = torch.fft.fft(b)
        return torch.fft.ifft(fa * fb).real.float()

    def inverse_convolution(self, bound: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Unbinding = convolve with complex conjugate of b"""
        fb = torch.fft.fft(b)
        fbound = torch.fft.fft(bound)
        return torch.fft.ifft(fbound * fb.conj()).real.float()

    def encode_sequence(self, sequence: list[str]) -> torch.Tensor:
        composite = torch.zeros(self.dim)
        for i, word in enumerate(sequence):
            if word not in self.item_memory or i >= len(self.pos_memory):
                continue
            bound = self.circular_convolution(self.item_memory[word], self.pos_memory[i])
            composite += bound
        return composite / (torch.norm(composite) + 1e-8)

    def _beam_decode(self, vector: torch.Tensor, max_length: int = 12, sim_threshold: float = 0.12):
        """Beam search geometric decoding with inverse convolution"""
        beam = [(vector.clone(), [])]  # (residual, sequence)

        for pos in range(max_length):
            new_beam = []
            pos_v = self.pos_memory[pos]

            for residual, seq in beam:
                # Unbind current position
                unbound = self.inverse_convolution(residual, pos_v)
                unbound /= torch.norm(unbound) + 1e-8

                # Score all items
                sims = torch.matmul(torch.stack(list(self.item_memory.values())), unbound)
                vals, idxs = torch.topk(sims, k=min(self.beam_width, len(self.vocab)))

                for val, idx in zip(vals, idxs):
                    if val < sim_threshold:
                        continue
                    word = self.vocab[idx]
                    bound = self.circular_convolution(self.item_memory[word], pos_v)
                    new_residual = residual - val * bound
                    new_residual /= torch.norm(new_residual) + 1e-8
                    new_seq = seq + [word]
                    new_beam.append((new_residual, new_seq))

            if not new_beam:
                break

            # Keep top beam_width
            new_beam.sort(key=lambda x: len(x[1]), reverse=True)
            beam = new_beam[:self.beam_width]

        if beam:
            return max(beam, key=lambda x: len(x[1]))[1]
        return []

    def map_decode(self, composite: torch.Tensor, max_length: int = 12, sim_threshold: float = 0.12):
        """Kleyko MAP: multiple noisy copies + majority vote per position"""
        votes = [[] for _ in range(max_length)]

        for _ in range(self.num_noisy_copies):
            noisy = composite + torch.randn_like(composite) * self.noise_level
            noisy /= torch.norm(noisy) + 1e-8
            decoded = self._beam_decode(noisy, max_length, sim_threshold)
            for pos, word in enumerate(decoded):
                if pos < max_length:
                    votes[pos].append(word)

        # Majority vote
        result = []
        for pos_votes in votes:
            if not pos_votes:
                break
            most_common = Counter(pos_votes).most_common(1)
            result.append(most_common[0][0])

        return result

    def generate(self, seed_text="", max_length=15, creativity=0.25):
        if seed_text:
            seq = seed_text.split()
        else:
            seq = [random.choice(self.vocab)]

        composite = self.encode_sequence(seq)
        generated = list(seq)

        seen = set(seq)  # avoid early repetition

        for _ in range(max_length):
            decoded = self.map_decode(composite, max_length=1)
            if not decoded:
                break

            next_word = decoded[0]

            # Creativity injection
            if random.random() < creativity:
                alt = self.map_decode(composite, max_length=3)
                if len(alt) > 1:
                    next_word = alt[1]

            if next_word in seen and len(generated) > 5:
                # Avoid repetition after a while
                next_word = random.choice(self.vocab)

            generated.append(next_word)
            seen.add(next_word)

            pos = len(generated) - 1
            if pos >= len(self.pos_memory):
                break

            bound_new = self.circular_convolution(self.item_memory[next_word], self.pos_memory[pos])
            composite += bound_new
            composite /= torch.norm(composite) + 1e-8

        return " ".join(generated)

# ──────────────────────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vocabulary = [
        "జ్ఞానం", "నది", "ప్రవహిస్తుంది", "knowledge", "flows", "river",
        "wisdom", "mind", "truth", "peace", "silence", "reflection",
        "Force", "=", "Mass", "times", "Acceleration", "Energy",
        "code", "python", "function", "return"
    ]

    model = PureHDCTextGen(
        vocabulary,
        dim=4096,
        beam_width=4,
        num_noisy_copies=7
    )

    print("=== RECONSTRUCTION TEST ===")
    seq = ["Force", "=", "Mass", "times", "Acceleration"]
    encoded = model.encode_sequence(seq)
    beam_decoded = model._beam_decode(encoded, max_length=8)
    print("Original :", " ".join(seq))
    print("Beam decoded :", " ".join(beam_decoded))

    print("\n=== KLEYKO MAP + NOISY COPIES ===")
    map_decoded = model.map_decode(encoded, max_length=8)
    print("MAP decoded :", " ".join(map_decoded))

    print("\n=== AUTOREGRESSIVE GENERATION ===")
    print("From seed 'జ్ఞానం నది':")
    print(model.generate("జ్ఞానం నది", max_length=18, creativity=0.3))

    print("\nFrom seed 'Force =':")
    print(model.generate("Force =", max_length=12, creativity=0.25))
