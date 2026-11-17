# 🔬 Universal STEM Master Scientific Encyclopedia & Reference Compendium

This authoritative compendium establishes foundational, intermediate, and advanced scientific grounding across **Mathematics**, **Physics**, **Chemistry**, and **Biology**.

---

## 1. MATHEMATICS (Pure & Applied)

### 1.1 Differential & Integral Calculus
- **Fundamental Theorem of Calculus (FTC)**:
  - Part 1: If $f$ is continuous on $[a, b]$ and $F(x) = \int_a^x f(t)\,dt$, then $F'(x) = f(x)$ for all $x \in (a, b)$.
  - Part 2: If $f$ is continuous on $[a, b]$ and $F$ is an antiderivative of $f$, then $\int_a^b f(x)\,dx = F(b) - F(a)$.
- **Taylor & Maclaurin Series**:
  $$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n$$
  - $e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots$
  - $\sin(x) = \sum_{n=0}^{\infty} (-1)^n \frac{x^{2n+1}}{(2n+1)!} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \dots$
  - $\cos(x) = \sum_{n=0}^{\infty} (-1)^n \frac{x^{2n}}{(2n)!} = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \dots$
  - $\ln(1+x) = \sum_{n=1}^{\infty} (-1)^{n+1} \frac{x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \dots \quad \text{for } |x| < 1$
- **Multivariable Calculus Theorems**:
  - **Green's Theorem**: $\oint_{\partial D} (L\,dx + M\,dy) = \iint_D \left(\frac{\partial M}{\partial x} - \frac{\partial L}{\partial y}\right) dx\,dy$
  - **Stokes' Theorem**: $\oint_{\partial \Sigma} \mathbf{F} \cdot d\mathbf{r} = \iint_\Sigma (\nabla \times \mathbf{F}) \cdot d\mathbf{\Sigma}$
  - **Divergence (Gauss's) Theorem**: $\oiint_{\partial V} \mathbf{F} \cdot d\mathbf{S} = \iiint_V (\nabla \cdot \mathbf{F})\,dV$

### 1.2 Linear Algebra & Matrix Theory
- **Eigenvalues & Eigenvectors**: For linear operator $A \in \mathbb{R}^{n \times n}$, non-zero $v \in \mathbb{R}^n$ satisfies $Av = \lambda v$, where $\det(A - \lambda I) = 0$ is the characteristic polynomial.
- **Spectral Theorem**: Any real symmetric matrix $A$ is orthogonally diagonalizable: $A = Q \Lambda Q^T$, where $Q$ is orthogonal ($Q^T Q = I$) and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$ contains real eigenvalues.
- **Singular Value Decomposition (SVD)**: For any $M \in \mathbb{R}^{m \times n}$, $M = U \Sigma V^T$, where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal, and $\Sigma \in \mathbb{R}^{m \times n}$ contains non-negative singular values $\sigma_i = \sqrt{\lambda_i(M^T M)}$ in descending order.

### 1.3 Abstract Algebra & Number Theory
- **Euler's Totient Theorem**: If $\gcd(a, n) = 1$, then $a^{\phi(n)} \equiv 1 \pmod n$. For prime $p$, Fermat's Little Theorem holds: $a^{p-1} \equiv 1 \pmod p$.
- **Chinese Remainder Theorem (CRT)**: If pairwise coprime integers $n_1, n_2, \dots, n_k$ are given, the system $x \equiv a_i \pmod{n_i}$ has a unique solution modulo $N = \prod n_i$.
- **Infinitude of Primes (Euclid's Proof)**: Assume finitely many primes $p_1, p_2, \dots, p_k$. Form integer $N = p_1 p_2 \dots p_k + 1$. Either $N$ is prime, or it has a prime factor $q$. If $q = p_i$ for some $i$, then $q$ divides $N - p_1 \dots p_k = 1$, contradiction. Thus infinitely many primes exist.

### 1.4 Differential Equations
- **First-Order Linear ODE**: $\frac{dy}{dx} + P(x)y = Q(x)$. Integrating factor $I(x) = \exp\left(\int P(x)\,dx\right) \implies y(x) = \frac{1}{I(x)}\left(\int I(x)Q(x)\,dx + C\right)$.
- **Second-Order Constant Coefficient Homogeneous ODE**: $a y'' + b y' + c y = 0$. Characteristic equation $a r^2 + b r + c = 0$:
  - Distinct real roots $r_1 \ne r_2$: $y = C_1 e^{r_1 x} + C_2 e^{r_2 x}$
  - Repeated root $r_1 = r_2 = r$: $y = (C_1 + C_2 x)e^{rx}$
  - Complex roots $\alpha \pm i\beta$: $y = e^{\alpha x}(C_1 \cos(\beta x) + C_2 \sin(\beta x))$
- **Laplace Transform**: $\mathcal{L}\{f(t)\} = F(s) = \int_0^{\infty} e^{-st} f(t)\,dt$. Key identities:
  - $\mathcal{L}\{e^{at}\} = \frac{1}{s - a}, \quad \mathcal{L}\{\sin(\omega t)\} = \frac{\omega}{s^2 + \omega^2}, \quad \mathcal{L}\{\cos(\omega t)\} = \frac{s}{s^2 + \omega^2}$
  - $\mathcal{L}\{f'(t)\} = s F(s) - f(0)$

---

## 2. PHYSICS (Classical to Modern)

### 2.1 Fundamental Physical Constants
- Speed of light in vacuum: $c = 299,792,458\text{ m/s}$ (exact)
- Planck's constant: $h = 6.62607015 \times 10^{-34}\text{ J}\cdot\text{s}$, $\hbar = \frac{h}{2\pi} = 1.054571817 \times 10^{-34}\text{ J}\cdot\text{s}$
- Elementary charge: $e = 1.602176634 \times 10^{-19}\text{ C}$
- Boltzmann constant: $k_B = 1.380649 \times 10^{-23}\text{ J/K}$
- Universal Gravitational constant: $G = 6.67430 \times 10^{-11}\text{ m}^3\text{kg}^{-1}\text{s}^{-2}$
- Avogadro constant: $N_A = 6.02214076 \times 10^{23}\text{ mol}^{-1}$
- Universal Gas constant: $R = N_A k_B = 8.314462618\text{ J}\cdot\text{mol}^{-1}\text{K}^{-1}$
- Permittivity of free space: $\varepsilon_0 \approx 8.8541878128 \times 10^{-12}\text{ F/m}$
- Permeability of free space: $\mu_0 \approx 1.256637062 \times 10^{-6}\text{ N/A}^2$ ($c = \frac{1}{\sqrt{\varepsilon_0 \mu_0}}$)

### 2.2 Classical Mechanics & Lagrangian Formalism
- **Newton's Laws of Motion**:
  1. Inertia: An object remains at rest or uniform velocity unless acted on by a net external force ($\sum \mathbf{F} = 0 \implies \frac{d\mathbf{v}}{dt} = 0$).
  2. Momentum flux: $\mathbf{F} = \frac{d\mathbf{p}}{dt} = m\mathbf{a}$ (for constant mass).
  3. Action-Reaction: $\mathbf{F}_{12} = -\mathbf{F}_{21}$.
- **Lagrangian Mechanics**:
  - Lagrangian $L = T - V$ (kinetic minus potential energy).
  - Euler-Lagrange equations derived from Hamilton's Principle of Least Action $\delta S = \delta \int L\,dt = 0$:
    $$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = 0$$
- **Hamiltonian Mechanics**:
  - Conjugate momentum $p_i = \frac{\partial L}{\partial \dot{q}_i}$. Hamiltonian $H(q, p, t) = \sum p_i \dot{q}_i - L = T + V$.
  - Canonical equations: $\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}$.

### 2.3 Electrodynamics & Maxwell's Equations
- **Maxwell's Equations in Differential Form (SI units)**:
  1. Gauss's Law for Electricity: $\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$ (electric field diverges from electric charge).
  2. Gauss's Law for Magnetism: $\nabla \cdot \mathbf{B} = 0$ (no magnetic monopoles).
  3. Faraday's Law of Induction: $\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$ (time-varying magnetic flux induces EMF).
  4. Ampère-Maxwell Law: $\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}$ (currents and displacement currents generate magnetic fields).
- **Electromagnetic Wave Equation in Vacuum**:
  $$\nabla^2 \mathbf{E} - \frac{1}{c^2} \frac{\partial^2 \mathbf{E}}{\partial t^2} = 0, \quad \nabla^2 \mathbf{B} - \frac{1}{c^2} \frac{\partial^2 \mathbf{B}}{\partial t^2} = 0, \quad c = \frac{1}{\sqrt{\mu_0 \varepsilon_0}}$$
- **Poynting Vector** (energy flux density): $\mathbf{S} = \frac{1}{\mu_0} (\mathbf{E} \times \mathbf{B})\text{ [W/m}^2\text{]}$.

### 2.4 Thermodynamics & Statistical Mechanics
- **Laws of Thermodynamics**:
  - 0th Law: Thermal equilibrium is transitive ($A \sim B$ and $B \sim C \implies A \sim C$), establishing temperature.
  - 1st Law: Conservation of energy: $dU = \delta Q - \delta W = T\,dS - P\,dV$.
  - 2nd Law: The total entropy of an isolated system never decreases: $\Delta S_{\text{isolated}} \ge 0$. Clausius definition: $dS = \frac{\delta Q_{\text{rev}}}{T}$.
  - 3rd Law: As $T \to 0\text{ K}$, entropy approaches a constant minimum value ($S \to 0$ for a perfect crystal).
- **Thermodynamic Potentials**:
  - Enthalpy: $H = U + PV \implies dH = T\,dS + V\,dP$
  - Helmholtz Free Energy: $F = U - TS \implies dF = -S\,dT - P\,dV$
  - Gibbs Free Energy: $G = H - TS \implies dG = -S\,dT + V\,dP$
- **Carnot Engine Efficiency**: $\eta_{\text{Carnot}} = 1 - \frac{T_C}{T_H}$.

### 2.5 Quantum Mechanics
- **Schrödinger Equation**:
  - Time-dependent: $i\hbar \frac{\partial}{\partial t}\Psi(\mathbf{r}, t) = \hat{H} \Psi(\mathbf{r}, t)$
  - Time-independent: $\hat{H} \psi(\mathbf{r}) = E \psi(\mathbf{r})$, where $\hat{H} = -\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r})$.
- **Heisenberg Uncertainty Principle**: $\sigma_x \sigma_p \ge \frac{\hbar}{2}$, $\Delta E \Delta t \ge \frac{\hbar}{2}$.
- **Quantum Harmonic Oscillator**: Potential $V(x) = \frac{1}{2}m\omega^2 x^2$. Eigenenergies $E_n = \left(n + \frac{1}{2}\right)\hbar\omega$, with ground-state zero-point energy $E_0 = \frac{1}{2}\hbar\omega$.

### 2.6 Special & General Relativity
- **Lorentz Transformation**: For relative motion along $x$ with velocity $v$, Lorentz factor $\gamma = \frac{1}{\sqrt{1 - v^2/c^2}}$:
  $$x' = \gamma(x - vt), \quad t' = \gamma\left(t - \frac{vx}{c^2}\right), \quad y' = y, \quad z' = z$$
- **Mass-Energy Equivalence**: $E^2 = (pc)^2 + (m_0 c^2)^2$. At rest ($p = 0$), $E = m_0 c^2$.
- **Einstein Field Equations (General Relativity)**:
  $$G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$$
  where $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2}R g_{\mu\nu}$ is the Einstein tensor, $R_{\mu\nu}$ is the Ricci curvature tensor, $R$ is scalar curvature, and $T_{\mu\nu}$ is the stress-energy-momentum tensor.

---

## 3. CHEMISTRY (Organic, Inorganic, Physical & Analytical)

### 3.1 Organic Chemistry & Reaction Mechanisms
- **Nucleophilic Substitution**:
  - **$S_N2$ Mechanism**: Bimolecular, concerted, single-step. Rate equation $v = k[\text{Substrate}][\text{Nu}^-]$. Requires accessible backside attack, resulting in **Walden inversion** of stereochemistry. Substrate reactivity order: Methyl > $1^\circ$ > $2^\circ$ >>> $3^\circ$ (steric hindrance prevents $3^\circ$). Favored by polar aprotic solvents (DMSO, DMF, acetone).
  - **$S_N1$ Mechanism**: Unimolecular, stepwise. Rate-determining step is heterolytic C-X bond cleavage forming a planar $sp^2$ carbocation intermediate ($v = k[\text{Substrate}]$). Nucleophile attacks from either face, yielding **racemization** (with slight inversion bias from ion-pair shielding). Substrate reactivity order: $3^\circ$ > $2^\circ$ >> $1^\circ$ (carbocation stability: allylic/benzylic $\approx 3^\circ > 2^\circ > 1^\circ$). Favored by polar protic solvents ($H_2O$, $CH_3OH$, $EtOH$).
- **Elimination Reactions**:
  - **E2 Mechanism**: Bimolecular, concerted. Base abstracts a $\beta$-proton while leaving group departs anti-periplanar (dihedral angle $180^\circ$). Regiochemistry follows **Zaitsev's rule** (more substituted, thermodynamically stable alkene) with unhindered bases (e.g. $EtO^-$) or **Hofmann's rule** with bulky bases (e.g. potassium $tert$-butoxide $t\text{-BuOK}$).
  - **E1 Mechanism**: Stepwise via carbocation intermediate; competes with $S_N1$ at elevated temperatures.
- **Electrophilic Aromatic Substitution (EAS)**:
  - Benzene reacts with electrophile $E^+$ through arenium ion ($\sigma$-complex, Wheland intermediate), followed by proton loss restoring aromatic sextet.
  - Activating/ortho-para directors: $-\text{OH}, -\text{OCH}_3, -\text{NH}_2, -\text{CH}_3$ (resonance electron donation stabilizes carbocation).
  - Deactivating/meta directors: $-\text{NO}_2, -\text{CN}, -\text{CHO}, -\text{COOH}, -\text{CF}_3$ (electron withdrawing groups destabilize ortho/para sigma complexes).
  - Halogens ($-\text{Cl}, -\text{Br}, -\text{I}$): Deactivating (inductive electron withdrawal) yet ortho/para directing (lone pair resonance donation).
- **Carbonyl Chemistry**:
  - **Aldol Condensation**: Enol or enolate ion attacks another carbonyl compound forming a $\beta$-hydroxyaldehyde/ketone, which upon heating dehydrates to an $\alpha,\beta$-unsaturated carbonyl compound.
  - **Diels-Alder Reaction**: $[4+2]$ cycloaddition between a conjugated cisoid diene and a dienophile (alkene/alkyne) producing a cyclohexene ring. Stereospecific, concerted, pericyclic reaction obeying Woodward-Hoffmann symmetry rules; yields endo diastereomer preferentially (secondary orbital interaction).

### 3.2 Physical Chemistry & Chemical Thermodynamics
- **Gibbs Free Energy & Spontaneity**:
  $$\Delta G = \Delta H - T\Delta S$$
  - $\Delta G < 0$: Spontaneous under constant $T, P$.
  - $\Delta G = 0$: Dynamic equilibrium.
  - $\Delta G > 0$: Non-spontaneous (endergonic).
  - Standard relation to equilibrium constant $K_{eq}$: $\Delta G^\circ = -RT \ln K_{eq} \implies K_{eq} = \exp\left(-\frac{\Delta G^\circ}{RT}\right)$.
- **Electrochemistry & The Nernst Equation**:
  - For half-cell or full reaction $aA + bB \rightleftharpoons cC + dD$ transferring $n$ moles of electrons:
    $$E = E^\circ - \frac{RT}{nF} \ln Q = E^\circ - \frac{0.05916\text{ V}}{n} \log_{10} Q \quad (\text{at } 298.15\text{ K})$$
    where $F = 96,485.332\text{ C/mol}$ (Faraday constant) and $Q = \frac{[C]^c [D]^d}{[A]^a [B]^b}$.
  - Relation to Gibbs free energy: $\Delta G^\circ = -nFE^\circ$.
- **Chemical Kinetics & The Arrhenius Equation**:
  $$k = A \exp\left(-\frac{E_a}{RT}\right) \iff \ln k = \ln A - \frac{E_a}{R}\left(\frac{1}{T}\right)$$
  where $k$ is rate constant, $A$ is pre-exponential frequency factor, and $E_a$ is activation energy.

### 3.3 Inorganic & Coordination Chemistry
- **Crystal Field Theory (CFT)**:
  - Transition metal $d$-orbitals ($d_{xy}, d_{yz}, d_{xz}, d_{x^2-y^2}, d_{z^2}$) split in ligand field.
  - **Octahedral Complex**: Ligands approach along Cartesian axes. Orbitals with lobes along axes ($d_{x^2-y^2}, d_{z^2}$) experience stronger repulsion and rise to $e_g$ level ($+0.6\Delta_o$). Inter-axial orbitals ($d_{xy}, d_{yz}, d_{xz}$) experience less repulsion and drop to $t_{2g}$ level ($-0.4\Delta_o$).
  - **Spectrochemical Series** (increasing field strength $\Delta_o$):
    $$I^- < Br^- < S^{2-} < SCN^- < Cl^- < F^- < OH^- < C_2O_4^{2-} < H_2O < NCS^- < edta^{4-} < NH_3 < en < NO_2^- < CN^- \approx CO$$
  - Strong-field ligands (e.g. $CN^-, CO$) cause large $\Delta_o > P$ (pairing energy), favoring low-spin configurations; weak-field ligands ($I^-, Cl^-$) yield high-spin configurations.

---

## 4. BIOLOGY (Molecular, Cellular, Physiological & Ecological)

### 4.1 Molecular Biology & The Central Dogma
- **DNA Structure & Replication**:
  - Double helix with anti-parallel strands ($5' \to 3'$ and $3' \to 5'$).
  - Complementary base pairing via hydrogen bonds: Adenine (A) pairs with Thymine (T) via 2 H-bonds; Guanine (G) pairs with Cytosine (C) via 3 H-bonds (yielding higher melting temperature $T_m$ for GC-rich sequences).
  - Semi-conservative replication: DNA Helicase unwinds helix; Single-Stranded Binding Proteins (SSBs) stabilize; Primase synthesizes RNA primer; DNA Polymerase III extends leading strand continuously ($5' \to 3'$) and lagging strand discontinuously via **Okazaki fragments**; DNA Polymerase I removes RNA primers and fills gaps; DNA Ligase seals phosphodiester nicks.
- **Transcription ($5' \to 3'$)**:
  - RNA Polymerase binds promoter (TATA box in eukaryotes, $-10$ Pribnow box in prokaryotes).
  - Pre-mRNA in eukaryotes undergoes $5'$ 7-methylguanosine capping, $3'$ polyadenylation (poly-A tail $\approx 200\text{ nt}$), and spliceosome-mediated removal of introns (lariat mechanism).
- **Translation**:
  - Genetic code: Triplet codons, degenerate (64 codons encode 20 amino acids + 3 stop codons: UAA, UAG, UGA; start codon AUG codes for Methionine).
  - Ribosome (70S prokaryotes: 50S + 30S; 80S eukaryotes: 60S + 40S) possesses A (aminoacyl), P (peptidyl), and E (exit) sites. Peptidyl transferase ribozyme in large ribosomal subunit forms peptide bonds.

### 4.2 Cellular Respiration & Bioenergetics
- **Complete Aerobic Respiration**:
  $$\text{C}_6\text{H}_{12}\text{O}_6 + 6\text{O}_2 \to 6\text{CO}_2 + 6\text{H}_2\text{O} + \approx 30 - 32\text{ ATP}$$
  1. **Glycolysis (Cytosol)**: Glucose ($6C$) $\to 2$ Pyruvate ($3C$) + $2\text{ ATP}$ (net, substrate-level phosphorylation) + $2\text{ NADH}$. Key regulatory enzyme: Phosphofructokinase-1 (PFK-1), allosterically inhibited by ATP and citrate, activated by AMP and fructose-2,6-bisphosphate.
  2. **Pyruvate Decarboxylation (Mitochondrial Matrix)**: $2\text{ Pyruvate} + 2\text{ CoA} + 2\text{ NAD}^+ \to 2\text{ Acetyl-CoA} + 2\text{ CO}_2 + 2\text{ NADH}$ via Pyruvate Dehydrogenase Complex (PDC).
  3. **Citric Acid (Krebs) Cycle (Mitochondrial Matrix)**: Acetyl-CoA ($2C$) condenses with Oxaloacetate ($4C$) $\to$ Citrate ($6C$). Per turn ($2$ turns per glucose): $3\text{ NADH} + 1\text{ FADH}_2 + 1\text{ GTP/ATP} + 2\text{ CO}_2$. Total per glucose: $6\text{ NADH} + 2\text{ FADH}_2 + 2\text{ ATP} + 4\text{ CO}_2$.
  4. **Oxidative Phosphorylation & Electron Transport Chain (Inner Mitochondrial Membrane)**:
     - Complex I (NADH dehydrogenase) pumps $4\text{ H}^+$ per NADH.
     - Complex II (Succinate dehydrogenase) transfers electrons from $\text{FADH}_2$ to ubiquinone (CoQ) without proton pumping.
     - Complex III (Cytochrome $bc_1$) pumps $4\text{ H}^+$.
     - Complex IV (Cytochrome $c$ oxidase) reduces $\frac{1}{2}\text{O}_2 + 2\text{H}^+ \to \text{H}_2\text{O}$ and pumps $2\text{ H}^+$.
     - Proton-motive force drives $\text{H}^+$ back through $F_o F_1$-ATP Synthase (chemiosmosis), generating $\approx 2.5\text{ ATP}$ per NADH and $\approx 1.5\text{ ATP}$ per $\text{FADH}_2$.

### 4.3 Enzyme Kinetics (Michaelis-Menten Model)
- Reaction scheme: $E + S \underset{k_{-1}}{\overset{k_1}{\rightleftharpoons}} ES \xrightarrow{k_{cat}} E + P$.
- Under steady-state approximation ($d[ES]/dt \approx 0$):
  $$v_0 = \frac{V_{\max}[S]}{K_M + [S]}$$
  where $V_{\max} = k_{cat}[E]_T$, and the Michaelis constant $K_M = \frac{k_{-1} + k_{cat}}{k_1}$ represents the substrate concentration at which initial velocity is half-maximal ($v_0 = V_{\max}/2$). Lower $K_M$ denotes higher substrate affinity.
- **Lineweaver-Burk Double Reciprocal Plot**:
  $$\frac{1}{v_0} = \frac{K_M}{V_{\max}}\left(\frac{1}{[S]}\right) + \frac{1}{V_{\max}}$$
  - Competitive Inhibition: $V_{\max}$ unchanged, apparent $K_M$ increases (intercept on $1/v_0$ axis invariant).
  - Non-competitive Inhibition: $V_{\max}$ decreases, $K_M$ unchanged (intercept on $1/[S]$ axis invariant).
  - Uncompetitive Inhibition: Both $V_{\max}$ and $K_M$ decrease by identical ratio (parallel lines).

### 4.4 Genetics & Inheritance
- **Mendelian Principles**:
  - Law of Segregation: Two alleles for each gene segregate during gamete formation so each gamete carries only one allele.
  - Law of Independent Assortment: Genes on non-homologous chromosomes assort independently during meiosis I (dihybrid cross phenotypic ratio $9:3:3:1$).
- **Hardy-Weinberg Equilibrium**: For a diploid population in absence of evolutionary forces (mutation, gene flow, genetic drift, natural selection, non-random mating):
  $$p + q = 1 \implies p^2 + 2pq + q^2 = 1$$
  where $p = f(A)$, $q = f(a)$, $p^2 = f(AA)$, $2pq = f(Aa)$, and $q^2 = f(aa)$.
