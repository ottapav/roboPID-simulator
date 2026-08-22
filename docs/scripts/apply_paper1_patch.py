#!/usr/bin/env python3
"""
Apply the scope clean-up patch to paper 1 (Model-Free PID Tuning by
Step-Response Inspection).

    python3 apply_paper1_patch.py main.tex [-o main_patched.tex]

Five edits:
  1  abstract      drop "inverse-response plants" from the covered classes
  2  Section 2     replace the membership paragraph by Lemma (lag chains)
  3  Section 5     battery membership becomes a one-line appeal to the lemma
  4  Limitations   name plant zeros in the exclusion, with the safe condition
  5  Introduction  add the parametric-region family + two references

Why edit 1/2 matter: the claim that $(1-\alpha s)/(s+1)^3$ satisfies
Assumption 1 "over its whole parameter range" is false. It holds only for
alpha <= sqrt(3) = 1.7321; above that the zero's rising magnitude beats the
three poles and |G| is no longer monotone.

Matching ignores differences in line wrapping and runs of whitespace, so it
survives reflowed source. Nothing is written unless EVERY edit matches exactly
once; on any failure the script reports which edit failed and exits non-zero,
leaving the input untouched.
"""
from __future__ import annotations
import argparse, re, sys, shutil
from pathlib import Path


def flex(pattern_text: str) -> re.Pattern:
    """Regex matching pattern_text with any whitespace run standing for any other."""
    parts = [re.escape(tok) for tok in pattern_text.split()]
    return re.compile(r"\s+".join(parts))


class Edit:
    def __init__(self, name, old, new, expect=1):
        self.name, self.old, self.new, self.expect = name, old, new, expect

    def apply(self, text):
        rx = flex(self.old)
        hits = list(rx.finditer(text))
        if len(hits) != self.expect:
            return None, f"matched {len(hits)} times, expected {self.expect}"
        return rx.sub(lambda m: self.new, text, count=self.expect), None


# --------------------------------------------------------------------------- 1
E1 = Edit(
    "1 abstract: drop inverse-response",
    r"""process with monotone gain and phase\fix{---first-order-plus-time-delay
models, higher-order lags and inverse-response plants among them---}and""",
    r"""process with monotone gain and phase\fix{---first-order-plus-time-delay
models and higher-order lag chains with dead time among them---}and""",
)

# --------------------------------------------------------------------------- 2
E2 = Edit(
    "2 Section 2: membership paragraph -> Lemma",
    r"""\fix{The familiar process classes satisfy it. First-order and
overdamped second-order models with time delay (FOPTD, SOPTD) do
for every dead time $L>0$, delay-free lag
chains do from four lags up, and mildly resonant plants do down to an
open-loop damping of $1/\sqrt2$, where the gain peak appears. Inverse
response is covered as well: a right-half-plane zero adds phase lag
while leaving $|G|$ monotone, so the family $(1-\alpha s)/(s+1)^3$ of
the test batch of \cite{AstromHagglund2004} satisfies the assumption
over its whole parameter range without any dead time. Of that batch,
only the integrating members are excluded by design; the delay-free
members of relative degree three sit exactly on the $-3\pi/2$ limit,
and any physical dead time restores them. The assumption is readable
from a measured frequency
response, so membership can be checked directly rather than argued from
a step-response shape \cite{SchlegelCech2005,Schlegel2002}.}""",
    r"""\fix{For the class that covers most regulatory loops, membership is decidable
by inspection.}
\begin{lemma}[Lag chains with dead time]\label{lem:class}
Let $G(s) = K\,e^{-Ls}\prod_{i=1}^{n}(\tau_i s+1)^{-1}$ with $K>0$,
$\tau_i>0$, $L\ge 0$. Then $|G|$ and $\arg G$ decrease strictly on
$(0,\infty)$, and $\arg G$ falls below $-3\pi/2$ iff $L>0$, or $L=0$ and
$n\ge 4$; such a plant satisfies Assumption~\ref{ass:plant} exactly when
$L>0$ or $n\ge 4$.
\end{lemma}

\begin{proof}
Differentiating, $\mathrm d\log|G|/\mathrm d\omega =
-\sum_i \tau_i^{2}\omega/(1+\tau_i^{2}\omega^{2})$ and
$\mathrm d(\arg G)/\mathrm d\omega =
-L-\sum_i \tau_i/(1+\tau_i^{2}\omega^{2})$, both sums of strictly negative
terms for $\omega>0$. As $\omega\to\infty$, $\arg G \sim -L\omega - n\pi/2$:
for $L>0$ this diverges and $-3\pi/2$ is crossed, while for $L=0$ each
$\arctan$ approaches $\pi/2$ from below, so $-n\pi/2$ is never attained and
$n\ge4$ is required. Stability and $|G(0)|<\infty$ follow from $\tau_i>0$.
\end{proof}

\fix{So FOPTD and SOPTD models qualify for every $L>0$ and delay-free lag
chains from four lags up; in the batch of \cite{AstromHagglund2004} the
delay-free members of relative degree three sit on the $-3\pi/2$ limit, which
any dead time restores. Three classes fall outside: complex pole pairs damped
below $1/\sqrt2$, which show a gain peak; integrating plants, which also
reverse the sign in Proposition~\ref{prop:mono}(i); and plants with finite
zeros, whose rising magnitude the poles must dominate.
Assumption~\ref{ass:plant} stays readable from a measured frequency response
\cite{SchlegelCech2005,Schlegel2002}, Lemma~\ref{lem:class} being the shortcut
when a parametric form is known.}""",
)

# --------------------------------------------------------------------------- 3
E3 = Edit(
    "3 Section 5: battery membership via the lemma",
    r"""All four plants satisfy Assumption~\ref{ass:plant}\fix{: each is a stable lag
chain with dead time, so gain and phase decrease strictly and the dead time
carries the phase past $-3\pi/2$}
(Table~\ref{tab:battery}).""",
    r"""\fix{All four plants are lag chains with dead time and $n\ge 4$, so
Lemma~\ref{lem:class} applies directly and Assumption~\ref{ass:plant} holds}
(Table~\ref{tab:battery}).""",
)

# --------------------------------------------------------------------------- 4
E4 = Edit(
    "4 Limitations: name plant zeros",
    r"""1)~\fix{Resonance strong enough to break the
monotone gain, or an integrating plant, falls outside
Assumption~\ref{ass:plant} and breaks this claim---visibly and safely,
by stopping the iteration at a bound.""",
    r"""1)~\fix{Resonance strong enough to break the monotone gain, an integrating
plant, or a plant zero slow enough to lift $|G|$, falls outside
Assumption~\ref{ass:plant} and breaks this claim---visibly and safely,
by stopping the iteration at a bound. For a single real zero of time constant
$\tau_z$ the monotone magnitude survives whenever $\tau_z\le\max_i\tau_i$,
which we do not pursue here.""",
)

# --------------------------------------------------------------------------- 5
E5a = Edit(
    "5a Introduction: two remedies -> three",
    r"""The literature offers two remedies, and both discard the inspection.""",
    r"""The literature offers three remedies, and all discard the inspection.""",
)

E5b = Edit(
    "5b Introduction: add the parametric-region family",
    r"""online perturbation of the gains
\cite{KillingsworthKrstic}.""",
    r"""online perturbation of the gains
\cite{KillingsworthKrstic}.
\fix{3)~Parametric-region methods compute the admissible set directly in the
controller's parameter plane, every setting meeting an $H_\infty$
specification \cite{BrabecSchlegel2023,Ho2003}: a region with a certificate
rather than one tuning, but needing a plant model, a weighting function and a
level $\gamma$ fixed in advance, for two free parameters at a time.}""",
)

E5c = Edit(
    "5c bibliography: two new entries",
    r"""\bibitem{KrausMyron1984} T.W.~Kraus, T.J.~Myron, Self-tuning PID""",
    r"""\bibitem{BrabecSchlegel2023} M.~Brabec, M.~Schlegel, Analytical design of a
wide class of controllers with two tunable parameters based on $H_\infty$
specifications, in: Proc. 24th Int. Conf. Process Control, 2023, pp.~221--226.
\bibitem{Ho2003} M.-T. Ho, Synthesis of $H_\infty$ PID controllers: a
parametric approach, Automatica 39 (6) (2003) 1069--1075.
\bibitem{KrausMyron1984} T.W.~Kraus, T.J.~Myron, Self-tuning PID""",
)


E6 = Edit(
    "6 bibliography: tighten list spacing",
    r"""\begingroup\small
\begin{thebibliography}{00}""",
    r"""\begingroup\footnotesize
\begin{thebibliography}{00}
\setlength{\itemsep}{0pt}\setlength{\parsep}{0pt}""",
)


E7 = Edit(
    "7 figures: 0.78 -> 0.70 textwidth (recovers the page the lemma costs)",
    r"""\includegraphics[width=0.78\textwidth]""",
    r"""\includegraphics[width=0.70\textwidth]""",
    expect=3,
)

EDITS = [E1, E2, E3, E4, E5a, E5b, E5c, E6, E7]


def ensure_lemma_env(text: str) -> tuple[str, str]:
    """Add \newtheorem{lemma} if absent (Edit 2 needs it)."""
    if re.search(r"\\newtheorem\{lemma\}", text):
        return text, "already present"
    anchor = re.search(r"\\newtheorem\{proposition\}\{Proposition\}", text)
    if not anchor:
        return text, "FAILED: no \\newtheorem{proposition} anchor found"
    i = anchor.end()
    return text[:i] + "\n\\newtheorem{lemma}{Lemma}" + text[i:], "inserted"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite source, keeping a .bak copy")
    a = ap.parse_args()

    text = a.source.read_text(encoding="utf-8")
    original = text

    text, lemma_note = ensure_lemma_env(text)
    if lemma_note.startswith("FAILED"):
        print(f"  preamble : {lemma_note}"); return 1
    print(f"  preamble : \\newtheorem{{lemma}} {lemma_note}")

    failures = []
    for e in EDITS:
        out, err = e.apply(text)
        if err:
            print(f"  FAIL  {e.name}: {err}")
            failures.append(e.name)
        else:
            text = out
            print(f"  ok    {e.name}")

    if failures:
        print("\nNo file written. Failing edits must be applied by hand;")
        print("see paper1_scope_patch.md for the exact OLD/NEW text.")
        return 1

    dest = a.source if a.in_place else (a.out or
                                        a.source.with_name(a.source.stem + "_patched.tex"))
    if a.in_place:
        shutil.copyfile(a.source, a.source.with_suffix(".tex.bak"))
        print(f"\n  backup   {a.source.with_suffix('.tex.bak').name}")
    dest.write_text(text, encoding="utf-8")
    print(f"  written  {dest}")
    print(f"  size     {len(original)} -> {len(text)} chars "
          f"({len(text)-len(original):+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
