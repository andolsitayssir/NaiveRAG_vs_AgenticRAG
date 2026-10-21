# Test Dataset (Medium) — UA-2 Délégation via Sourcing

**Pairs:** 6
**Tier:** medium — indirect phrasing, scenario-based, compound questions.

---

## UA-2-M-01

- **question:** J'ai besoin de confier une mission importante mais je n'ai personne en interne à qui la donner. Par où je commence ?
- **expected_ua:** UA-2
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** indirect, scenario
- **ground_truth:**
  > On sort du cadre UA-1 (mandataire identifié) pour entrer dans UA-2 (mandataire à trouver). Avant de lancer la recherche, il faut d'abord un upskilling sur la délégation, puis un Profiling du mandataire (compétences, crédibilité, appétence, posture attendues), puis une première recherche d'opportunités couvrant les cinq options : embauche, externalisation isocoût, prestataire externe, aide, aide temporaire débouchant sur une des trois précédentes. L'utilisateur choisit ensuite les options de sourcing à poursuivre avant la recherche finale.

## UA-2-M-02

- **question:** Un cabinet de conseil me propose de me fournir quelqu'un sans rien changer à mon budget actuel. C'est quel type de sourcing ?
- **expected_ua:** UA-2
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** indirect
- **ground_truth:**
  > C'est l'option d'« externalisation isocoût », l'une des cinq fonctions de sourcing prévues dans UA-2 : on remplace un coût interne existant par une prestation externe équivalente sans faire varier le budget global. À évaluer aux côtés des quatre autres options (embauche, prestataire externe, aide, aide temporaire convertible).

## UA-2-M-03

- **question:** Pourquoi il faut faire un profiling AVANT de chercher, et pas découvrir en cours de route ce qu'on veut ?
- **expected_ua:** UA-2
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** application
- **ground_truth:**
  > Parce que sans profiling les recherches sont dispersées et les cinq options de sourcing ne peuvent pas être évaluées de manière comparable. Le profiling fixe les compétences, la crédibilité, l'appétence et la posture attendues ; il permet aussi de détecter si la personne à trouver n'existe pas en interne (auquel cas on bascule de UA-1 à UA-2) et de cibler la recherche.

## UA-2-M-04

- **question:** J'hésite à embaucher car je ne sais pas encore si la charge sera pérenne. Quelle option de UA-2 puis-je préférer ?
- **expected_ua:** UA-2
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** scenario
- **ground_truth:**
  > L'aide temporaire qui débouchera sur une des trois premières solutions (embauche, externalisation isocoût, prestataire externe). C'est la cinquième option de sourcing, conçue pour permettre un test de la charge et du besoin avant d'engager une solution plus structurelle.

## UA-2-M-05

- **question:** Je croyais que ma collaboratrice X conviendrait, mais après qualification je vois que non. Quel est l'enchaînement recommandé ?
- **expected_ua:** UA-2
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** scenario, compound
- **ground_truth:**
  > Clôturer la qualification UA-1-I2 en constatant l'écart (par exemple sur l'appétence ou la posture), puis basculer dans UA-2 : refaire un profiling à partir des besoins objectivés, lancer la recherche d'opportunités sur les cinq options de sourcing, valider avec l'utilisateur les options à poursuivre, puis exécuter la fonction de Recrutement ou de Sourcing Externe sur les options retenues.

## UA-2-M-06

- **question:** Comment Viito doit-il me redonner la main entre la recherche d'opportunités et le recrutement ?
- **expected_ua:** UA-2
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** application
- **ground_truth:**
  > Après la première recherche d'opportunités sur les cinq types de sourcing, l'utilisateur doit sélectionner (étape [E] = edit) les options à retenir pour la recherche finale. Seulement alors les fonctions de Recrutement (pour l'embauche) ou de Sourcing Externe (pour les options externalisation, prestataire, aide) sont lancées. Cette remise-en-main évite que Viito engage des démarches lourdes sans arbitrage humain explicite.
