# Test Dataset (Advanced) — UA-2 Délégation via Sourcing

**Pairs:** 6
**Tier:** advanced — cross-UA reasoning, nuanced scenarios, adversarial phrasing.

---

## UA-2-A-01

- **question:** Quelle est la frontière fine entre « aide » et « aide temporaire qui débouchera sur une des trois premières solutions » ?
- **expected_ua:** UA-2
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** nuance, compare
- **ground_truth:**
  > « Aide » = solution durable dans laquelle quelqu'un vient épauler le manager sans que le dispositif soit conçu comme transitoire. « Aide temporaire qui débouchera… » = dispositif explicitement transitoire pour tester la charge, le besoin ou le profil, avant de basculer en embauche, externalisation isocoût ou prestataire externe. La différence n'est pas la durée en soi mais l'intention : l'aide temporaire est un test délibéré avec issue attendue parmi les trois solutions structurelles.

## UA-2-A-02

- **question:** Si je rate le profiling avant de lancer le sourcing, qu'est-ce que je risque concrètement ?
- **expected_ua:** UA-2|UA-1
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** case, cross_ua
- **ground_truth:**
  > Sans profiling précis : recherche dispersée, candidats incomparables entre les 5 options de sourcing, choix biaisé par la disponibilité plutôt que par l'adéquation. À l'arrivée, le mandataire trouvé peut ne pas satisfaire les trois critères UA-1 (compétence, volonté, compatibilité), et la délégation qui s'en suivra échouera — avec en plus un coût d'acquisition (embauche, prestation) déjà engagé. Le profiling est un investissement amont qui évite un échec aval coûteux.

## UA-2-A-03

- **question:** Pourquoi UA-2 prévoit une étape d'arbitrage utilisateur [E] entre l'exploration et l'exécution du sourcing ?
- **expected_ua:** UA-2
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** nuance
- **ground_truth:**
  > Parce que les cinq options de sourcing ont des implications très différentes (budget, juridique, engagement sur la durée, image interne) et que l'agent Viito ne peut pas décider seul quelle option privilégier. L'étape [E] formalise la remise-en-main utilisateur après exploration : celui-ci choisit les pistes à poursuivre, en cohérence avec ses contraintes non exprimées aux étapes [F]. Cela évite à Viito de lancer une recherche de recrutement ou de prestataire externe irréversible sur des critères qu'il aurait dû valider.

## UA-2-A-04

- **question:** J'ai identifié en UA-1 que la personne n'est pas le bon profil, mais elle en serait capable moyennant formation. Je reste en UA-1 ou je bascule UA-2 ?
- **expected_ua:** UA-1|UA-2
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** cross_ua, nuance
- **ground_truth:**
  > Dépend du critère manquant et du délai. Si ce n'est que la compétence technique qui manque et que la formation est réaliste dans le délai requis, rester en UA-1 avec un plan de formation (style persuasif/formatif) avant de lancer la délégation. Si la volonté ou la compatibilité manque, la formation ne suffira pas — bascule UA-2. Si la compétence manque *et* que le délai est court, bascule UA-2 pour ne pas « jeter à la mer pour voir si elle sait nager ».

## UA-2-A-05

- **question:** En quoi le choix entre embauche et externalisation isocoût n'est-il pas seulement financier ?
- **expected_ua:** UA-2
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** nuance, application
- **ground_truth:**
  > Au-delà du coût (équivalent par hypothèse en isocoût), les deux options diffèrent sur : l'intégration culturelle et la loyauté au projet collectif, la capacité à capitaliser l'apprentissage en interne, la souplesse de sortie, la perception par l'équipe (renfort « des nôtres » vs. prestataire extérieur), la maîtrise de l'information et des décisions. Le profiling doit intégrer ces dimensions et pas seulement la correspondance technique — faute de quoi le choix est réduit à un arbitrage comptable qui peut affaiblir la délégation.

## UA-2-A-06

- **question:** Si j'ai déjà un prestataire externe qui fait ce dont j'ai besoin, dois-je quand même passer par la méthode UA-2 ?
- **expected_ua:** UA-2
- **must_cite:** `Fiche Protocole UA-1 Délégation.docx`
- **tags:** adversarial, nuance
- **ground_truth:**
  > Oui pour les étapes amont (upskilling délégation, profiling, vérification que c'est bien la bonne solution). Même si le prestataire est déjà en place, il faut valider qu'il remplit les trois critères UA-1 (compétence, volonté d'assumer la mission spécifique, compatibilité), qu'il y a bien matière à déléguer un *processus complet* et non de simples tâches, et qu'un contrat de délégation sans ambiguïté est possible. Sauter ces étapes au nom du « on l'a déjà » est la recette d'une délégation qui s'enlisera.
