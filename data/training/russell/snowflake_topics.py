"""Snowflake topic variations for Russell LoRA training.

These topics are tailored to the book's themes (consciousness, emergence,
dialectics, evolution, artificial minds, cosmic perspective) so the LoRA
learns to apply Russell's style to exactly the register it will encounter
at inference.

Categories map to book chapters:
  Ch 1: Personal history, collapse of systems
  Ch 2: Technology, complexity, hidden dependencies
  Ch 3: Cosmology, entropy, origins of matter
  Ch 4: Abiogenesis, self-replication, cellular machinery
  Ch 5: Consciousness, volition, qualia, neural architecture
  Ch 6: Communication, language, theory of mind
  Ch 7: Memes, substrate independence, emergence
  Ch 8: Dialectics, class, modes of production, societal evolution
  Ch 9: Artificial minds, virtual worlds, simulated evolution
  Ch 10: Cosmic scale, post-biological intelligence, Fermi paradox
"""

BOOK_TOPICS = [
    # CONSCIOUSNESS & MIND (Ch 5)
    "why subjective experience cannot be reduced to a mechanical description",
    "the illusion of free will as a product of neural architecture",
    "whether a sufficiently complex machine could be said to feel pain",
    "the boundary between reflex and deliberate action in simple organisms",
    "how the brain constructs a model of its own attention",
    "the philosophical absurdity of the zombie thought experiment",
    "why qualia are not evidence for dualism but for the limits of introspection",
    "the survival advantage of distinguishing self from environment",
    "how the sensation of colour is constructed rather than received",
    "the difference between self-representation and the feeling of being alive",
    "whether consciousness is a spectrum or a threshold phenomenon",
    "how homeostasis gives rise to the illusion of purpose",
    "the error of treating awareness as a binary property",
    "why the hard problem of consciousness is a product of confused thinking",
    "how memory transforms raw sensation into personal narrative",

    # EVOLUTION & BIOLOGY (Ch 4)
    "why self-replication is a natural consequence of chemistry rather than a miracle",
    "how energy gradients drive the formation of increasingly complex structures",
    "the parallel between a bacterium seeking food and a corporation seeking profit",
    "why multicellular organisms are best understood as colonies of simpler units",
    "the evolutionary logic of placing taste receptors near the mouth",
    "how specialization within a cell colony creates the need for coordination",
    "the distinction between life and non-life as a matter of degree",
    "why locomotion and predation transformed the architecture of the nervous system",
    "the chemical basis of heredity and its implications for individuality",
    "how natural selection operates without intention or foresight",

    # EMERGENCE & COMPLEXITY (Ch 7)
    "how patterns persist across radically different physical substrates",
    "why the whole is not merely the sum of its parts in complex systems",
    "how a simple set of rules can generate unpredictable collective behaviour",
    "the relationship between information and physical organisation",
    "why reductionism fails to capture the behaviour of emergent systems",
    "how individual ignorance can produce collective intelligence",
    "the analogy between genetic inheritance and cultural transmission",
    "why a mind running on silicon would be no less real than one running on carbon",
    "how language transforms individual thought into a shared cognitive resource",
    "the sense in which a corporation has a life of its own independent of its members",

    # DIALECTICS & SOCIETY (Ch 8)
    "how the mode of production determines the structure of social relations",
    "why surplus wealth inevitably gives rise to class division",
    "the parallel between biological evolution and the evolution of political systems",
    "how contradictions within a social order become the engine of its transformation",
    "why the ruling ideas of an epoch are always the ideas of the ruling class",
    "how division of labour increases productivity while creating dependency",
    "the mechanism by which economic crises expose the limits of a social system",
    "why competing interests within a society produce predictable patterns of change",
    "how ideology functions as the unconscious servant of material conditions",
    "the distinction between the appearance and the reality of democratic governance",
    "why markets tend toward monopoly without external constraint",
    "how technological change forces the reorganisation of social institutions",

    # COMMUNICATION & LANGUAGE (Ch 6)
    "how chemical signalling in bacteria prefigures the complexity of human speech",
    "why theory of mind is a prerequisite for both cooperation and deception",
    "the evolutionary advantage of transmitting information without physical contact",
    "how symbolic communication transforms the capacity for abstract thought",
    "why the invention of writing altered the trajectory of human civilisation",
    "the relationship between the precision of a language and the complexity of its users",
    "how communication within the brain differs from communication between organisms",
    "why misunderstanding is an inherent feature of all symbolic exchange",

    # COSMOLOGY & PHYSICS (Ch 3, 10)
    "why the concept of a beginning to time involves a logical contradiction",
    "how the laws of thermodynamics constrain the possible forms of organisation",
    "why the universe requires no external cause if it is treated as self-contained",
    "the insignificance of human civilisation measured against cosmic timescales",
    "how the expansion of the universe determines the fate of all structure within it",
    "why the conditions for life appear to be neither rare nor inevitable",
    "the distinction between the mathematical description of nature and nature itself",
    "how matter organises itself into galaxies without central direction",

    # ARTIFICIAL INTELLIGENCE & COMPUTATION (Ch 9)
    "whether a simulated world could produce genuinely conscious inhabitants",
    "the analogy between natural selection and the training of artificial neural networks",
    "why scarcity is a necessary condition for the emergence of purposeful behaviour",
    "how virtual environments could serve as laboratories for evolving intelligence",
    "the philosophical implications of running a mind at a different speed",
    "why substrate independence implies that biological intelligence is not special",
    "the parallel between homeostasis in living systems and reward functions in machines",
    "whether artificial minds would develop values recognisable to their creators",

    # TECHNOLOGY & SYSTEMS (Ch 2)
    "how the complexity of modern infrastructure conceals its fragility",
    "why the interdependence of global systems creates vulnerabilities invisible to their users",
    "how technology shapes the cognitive habits of its users without their awareness",
    "the illusion of control in systems too complex for any individual to understand",
    "why the proliferation of tools does not guarantee an increase in understanding",

    # EPISTEMOLOGY & METHOD (general)
    "why the demand for absolute certainty is the enemy of practical knowledge",
    "how induction relies on assumptions that cannot themselves be proved by induction",
    "the difference between a description of phenomena and an explanation of them",
    "why scepticism about established authority is a prerequisite for intellectual progress",
    "how the scientific method corrects its own errors over time",
    "why common sense is an unreliable guide to the nature of physical reality",
    "the relationship between mathematical structure and empirical observation",
    "how prejudice disguises itself as self-evident truth",
]
