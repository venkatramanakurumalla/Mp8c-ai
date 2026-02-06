import torch
import random
from collections import Counter

# ============================================================
# PURE HDC / HRR TEXT GENERATOR (GodNode-SAFE)
# ============================================================

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

        # ----------------------------------------------------
        # Item Memory (Bipolar Unit Vectors)
        # ----------------------------------------------------
        self.item_memory = {}
        for w in self.vocab:
            v = torch.randn(dim).sign().float()
            v /= torch.norm(v) + 1e-8
            self.item_memory[w] = v

        # Vectorized cleanup memory
        self.vocab_matrix = torch.stack([self.item_memory[w] for w in self.vocab])

        # ----------------------------------------------------
        # Fixed Positional Memory (CRITICAL)
        # ----------------------------------------------------
        self.pos_memory = torch.randn(max_positions, dim).sign().float()
        self.pos_memory /= torch.norm(self.pos_memory, dim=1, keepdim=True) + 1e-8

    # ========================================================
    # HRR OPERATIONS
    # ========================================================

    def circular_convolution(self, a, b):
        return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b)).real.float()

    def inverse_convolution(self, bound, b):
        return torch.fft.ifft(
            torch.fft.fft(bound) * torch.conj(torch.fft.fft(b))
        ).real.float()

    # ========================================================
    # ENCODING
    # ========================================================

    def encode_sequence(self, sequence):
        composite = torch.zeros(self.dim)
        for i, word in enumerate(sequence):
            if word not in self.item_memory or i >= len(self.pos_memory):
                continue
            composite += self.circular_convolution(
                self.item_memory[word], self.pos_memory[i]
            )
        return composite / (torch.norm(composite) + 1e-8)

    # ========================================================
    # BEAM DECODER
    # ========================================================

    def _beam_decode(self, vector, max_length=12, sim_threshold=0.12):
        beam = [(vector.clone(), [])]

        for pos in range(max_length):
            new_beam = []
            pos_v = self.pos_memory[pos]

            for residual, seq in beam:
                unbound = self.inverse_convolution(residual, pos_v)
                unbound /= torch.norm(unbound) + 1e-8

                sims = torch.matmul(self.vocab_matrix, unbound)
                vals, idxs = torch.topk(
                    sims, k=min(self.beam_width, len(self.vocab))
                )

                for sim, idx in zip(vals, idxs):
                    if sim < sim_threshold:
                        continue

                    word = self.vocab[idx]
                    bound = self.circular_convolution(
                        self.item_memory[word], pos_v
                    )

                    new_residual = residual - sim * bound
                    new_residual /= torch.norm(new_residual) + 1e-8

                    new_beam.append((new_residual, seq + [word]))

            if not new_beam:
                break

            beam = sorted(new_beam, key=lambda x: len(x[1]), reverse=True)[:self.beam_width]

        return max(beam, key=lambda x: len(x[1]))[1] if beam else []

    # ========================================================
    # MAP DECODING (NOISE ROBUST)
    # ========================================================

    def map_decode(self, composite, max_length=12, sim_threshold=0.12):
        votes = [[] for _ in range(max_length)]

        for _ in range(self.num_noisy_copies):
            noisy = composite + torch.randn_like(composite) * self.noise_level
            noisy /= torch.norm(noisy) + 1e-8

            decoded = self._beam_decode(noisy, max_length, sim_threshold)
            for i, w in enumerate(decoded):
                votes[i].append(w)

        result = []
        for v in votes:
            if not v:
                break
            result.append(Counter(v).most_common(1)[0][0])

        return result

    # ========================================================
    # AUTOREGRESSIVE GENERATION
    # ========================================================

    def generate(self, seed_text="", max_length=15, creativity=0.25):
        seq = seed_text.split() if seed_text else [random.choice(self.vocab)]
        composite = self.encode_sequence(seq)
        generated = list(seq)
        seen = set(seq)

        for _ in range(max_length):
            decoded = self.map_decode(composite, max_length=1)
            if not decoded:
                break

            next_word = decoded[0]

            if random.random() < creativity:
                next_word = random.choice(self.vocab)

            if next_word in seen and len(generated) > 5:
                next_word = random.choice(self.vocab)

            generated.append(next_word)
            seen.add(next_word)

            pos = len(generated) - 1
            if pos >= len(self.pos_memory):
                break

            composite += self.circular_convolution(
                self.item_memory[next_word], self.pos_memory[pos]
            )
            composite /= torch.norm(composite) + 1e-8

        return " ".join(generated)
