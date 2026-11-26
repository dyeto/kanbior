faire une recherche pousssée pour faire un etat de l'art sur les models de tts fon, essentiel les models potentiels avec licence commercialisable qu'on pourrait fintime, faut aussi conclure et justeifier s'il vaut mieux entraimer à partir de 0 ou fine tuner, en gros l'état de l'art en faisont compromis entre precision et ressource pour inference
-------
# **État de l'art — Synthèse vocale (TTS) pour le Fon**

*Projet Kanbior · Novembre 2026*

## **1. Objectif de la recherche**

L'objectif est d'identifier la meilleure approche pour développer **kanbior-fon-tts-v1**, un modèle de synthèse vocale capable de transformer un texte en Fon (Fongbe) en une parole naturelle, intelligible et fidèle aux tonalités de la langue, tout en étant exploitable commercialement.

Contrairement à une recherche classique sur les performances des modèles TTS, quatre critères doivent être considérés simultanément :

- **Qualité linguistique** : prononciation, respect des tons, intelligibilité.
- **Qualité acoustique** : naturalité de la voix, prosodie, fluidité.
- **Ressources d'inférence** : consommation RAM, latence, taille du modèle, possibilité de fonctionner sur CPU.
- **Viabilité commerciale** : licence des poids, du code, des données et possibilité de fine-tuning.

Le précédent état de l'art de Kanbior proposait d'entraîner VITS à partir de zéro, principalement en raison des restrictions commerciales de Meta MMS-TTS. Cette approche mérite d'être réévaluée en considérant les autres modèles préentraînés disponibles.&#x20;

Kanbior state of the art for Fon speech AI.pdf



Le fait qu'un modèle ne supporte pas encore le Fon ne doit pas automatiquement l'exclure : sa capacité à apprendre une nouvelle langue par transfert d'apprentissage est précisément l'un des éléments à étudier.

## **2. État des modèles TTS existants pour le Fon**

### **2.1. Les modèles directement liés au Fon et aux langues africaines**

La recherche fait ressortir plusieurs initiatives pertinentes, dont certaines très récentes. Cependant, les restrictions de licence restent un obstacle majeur.

| Modèle                                                                                            | Architecture               | Licence des poids | Commercialisation |
| ------------------------------------------------------------------------------------------------- | -------------------------- | ----------------- | ----------------- |
| [MMS-TTS-Fon](https://huggingface.co/facebook/mms-tts-fon?utm_source=chatgpt.com)                 | VITS                       | CC-BY-NC-4.0      | Non               |
| [fon_tts_speaker_0](https://huggingface.co/jonathansuru/fon_tts_speaker_0?utm_source=chatgpt.com) | VITS                       | Non clarifiée     | Non validée       |
| [Afri-VITS](https://github.com/AfriSpeech/afri-vits-finetuner?utm_source=chatgpt.com)             | VITS multilingue adaptable | CC-BY-NC-4.0      | Non               |
| [Africa TTS](https://github.com/AfriSpeech/africa-tts?utm_source=chatgpt.com)                     | Pocket TTS                 | CC-BY-NC-4.0      | Non               |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice?utm_source=chatgpt.com)                       | Diffusion Language Model   | CC-BY-NC          | Non               |

Sources : documentation des modèles et dépôts officiels.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+4



### **2.2. Meta MMS-TTS-Fon**

Ce modèle constitue une référence naturelle pour évaluer Kanbior.

- Architecture VITS.
- Environ 36 millions de paramètres.
- Support natif du Fon.
- Inférence relativement légère.
- Modèle spécialisé dans une seule langue.
- Préentraînement principalement issu de lectures bibliques dans le cadre du projet MMS.

Son principal avantage est qu'il possède déjà une connaissance acoustique et phonétique du Fon.

Son principal inconvénient est sa licence non commerciale, qui empêche de retenir ses poids comme base d'un produit commercial sans autorisation supplémentaire.

**Il faudra néanmoins l'utiliser comme baseline scientifique**, dans un cadre compatible avec sa licence.

### **2.3. Africa TTS : une initiative particulièrement intéressante**

Le projet AfriSpeech a développé un modèle basé sur Pocket TTS, couvrant 92 langues africaines ainsi que l'anglais.

Les performances annoncées par ses auteurs sont particulièrement intéressantes pour Kanbior :

Langues africaines entraînées

# **92**

Vitesse sur 4 vCPU

# **2,4×**

Plus rapide que le temps réel

Taille du téléchargement

# **440 Mo**

Comparaison avec MMS-TTS

# **21 / 27**

Langues avec une meilleure intelligibilité mesurée

Le projet utilise notamment des représentations phonétiques IPA, permet le streaming et propose un déploiement mobile via ONNX.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



Malheureusement, ses poids sont non commerciaux. Je n'ai également pas pu confirmer que le Fon figure parmi les 92 langues effectivement entraînées.

En revanche, cette initiative démontre la faisabilité d'une approche très intéressante : **adapter un modèle TTS compact et préentraîné aux langues africaines, sans dépendre d'un GPU en production.**

Il serait pertinent de reproduire cette démarche à partir des poids originaux de Pocket TTS, plutôt que de ceux d'Africa TTS.

## **3. Modèles préentraînés potentiellement commercialisables**

Le périmètre de recherche doit être élargi aux modèles ne supportant pas encore le Fon, mais susceptibles de l'apprendre.

### **3.1. Comparatif général**

Les tailles ci-dessous sont approximatives. Elles ne représentent pas nécessairement la totalité du pipeline d'inférence, notamment lorsqu'un vocodeur externe est nécessaire.

| Modèle              | Paramètres | Licence indicative des poids     | Inférence cible | Priorité Kanbior |
| ------------------- | ---------- | -------------------------------- | --------------- | ---------------- |
| Piper / VITS Medium | \~15 M     | Selon checkpoint, CC0 disponible | CPU / Mobile    | Très haute       |
| MeloTTS             | \~52 M     | MIT pour certains checkpoints    | CPU             | Moyenne          |
| Kokoro              | 82 M       | Apache 2.0                       | CPU             | Moyenne          |
| Pocket TTS          | 100 M      | CC-BY-4.0                        | CPU / Mobile    | Très haute       |
| SpeechT5            | \~144 M    | MIT                              | CPU / GPU       | Moyenne          |
| Matcha-TTS          | Variable   | MIT pour les poids identifiés    | CPU / GPU       | Haute            |
| ToucanTTS           | Variable   | Apache 2.0 affichée              | CPU / GPU       | À investiguer    |
| MagpieTTS           | \~364 M    | NVIDIA Open Model License        | GPU privilégié  | Moyenne          |
| CosyVoice           | \~500 M    | Apache 2.0 selon version         | GPU privilégié  | Moyenne          |
| Qwen3-TTS           | 600 M      | Apache 2.0                       | GPU privilégié  | Haute            |
| Qwen3-TTS Large     | 1 700 M    | Apache 2.0                       | GPU             | Faible           |

Sources : fiches des modèles, documentation des auteurs et comparatifs de novembre 2026.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.assemblyai.com\&sz=32)

AssemblyAI+5



### **Comparaison de la taille des architectures**

Nombre approximatif de paramètres — hors composants externes

Piper Medium

15 M

MMS-TTS

36 M

MeloTTS

52 M

Kokoro

82 M

Pocket TTS

100 M

SpeechT5

144 M

MagpieTTS

364 M

CosyVoice

500 M

Qwen3-TTS 0.6B

600 M

Qwen3-TTS 1.7B

1 700 M

En vert : les deux candidats prioritaires pour une inférence économique. La taille ne constitue pas une mesure directe de la qualité ou de la latence.

## **4. Analyse approfondie des meilleurs candidats**

### **4.1. Piper / VITS — le meilleur candidat pour une inférence minimale**

Candidat prioritaire

\~15 M de paramètres

Piper est une implémentation optimisée de VITS, conçue principalement pour l'exécution locale et rapide sur CPU.

VITS combine dans une même architecture l'apprentissage des représentations acoustiques, l'alignement texte-audio, la modélisation des durées et la génération de la forme d'onde.

Architecture d'inférence VITS simplifiée.

**Avantages**

- Très faible consommation de ressources.
- Export ONNX mature.
- Fonctionnement sans GPU.
- Entraînement mono ou multi-locuteur.
- Fine-tuning interlangue explicitement documenté.
- Architecture non autorégressive adaptée aux réponses vocales courtes.

La documentation Piper recommande d'ailleurs de partir d'un checkpoint existant, même lorsque sa langue d'origine est différente.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



#### **Un checkpoint commercial particulièrement intéressant**

[Thorsten-Voice/Piper-Scratch](https://huggingface.co/Thorsten-Voice/Piper-Scratch?utm_source=chatgpt.com)

Ce checkpoint publié en novembre 2026 présente un intérêt majeur :

| Caractéristique                   | Valeur          |
| --------------------------------- | --------------- |
| Langue initiale                   | Allemand        |
| Licence des poids                 | CC0             |
| Architecture                      | Piper Medium    |
| Fréquence audio                   | 22,05 kHz       |
| Données initiales                 | 11,22 heures    |
| Checkpoint d'entraînement complet | Disponible      |
| Données non commerciales héritées | Aucune déclarée |

Le modèle a été entraîné sans checkpoint tiers, puis publié avec ses poids complets pour permettre le fine-tuning.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



Un autre checkpoint intéressant est [Piper Thorsten Medium](https://huggingface.co/datasets/rhasspy/piper-checkpoints/blob/main/de/de_DE/thorsten/medium/MODEL_CARD?utm_source=chatgpt.com), dont la version récente repose sur une base flamande CC0.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

rhasspy/piper-checkpoints at main



**Attention :** il ne faut pas sélectionner n'importe quelle voix Piper. Certains anciens checkpoints héritent de données Lessac réservées à la recherche.

#### **Limites pour le Fon**

Le modèle ne connaît pas nécessairement :

- Les phonèmes spécifiques au Fon.
- Les contrastes tonals lexicaux.
- Les règles propres aux voyelles nasales.
- Les relations entre accents graphiques et tons prononcés.

Il faudra donc adapter le traitement phonétique et les embeddings associés, puis probablement réaliser un fine-tuning complet plutôt qu'un simple LoRA.

**Verdict : excellent candidat pour la version de production la plus légère.**

### **4.2. Pocket TTS — le meilleur compromis potentiel entre qualité et ressources**

Candidat prioritaire

100 M de paramètres

Pocket TTS est développé par Kyutai avec l'objectif de proposer une synthèse vocale de haute qualité directement sur CPU.

Il apporte des capacités généralement associées à des modèles beaucoup plus lourds.

Paramètres

# **100 M**

Latence initiale annoncée

# **\~200 ms**

Vitesse sur MacBook Air M4

# **\~6×**

Temps réel

CPU utilisé dans ce benchmark

# **2 cœurs**

Ces chiffres proviennent des auteurs et concernent les modèles d'origine, pas une version fine-tunée en Fon.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



| Caractéristique             | Disponibilité            |
| --------------------------- | ------------------------ |
| Streaming audio             | Oui                      |
| Inférence CPU               | Oui                      |
| Fine-tuning / entraînement  | Code publié en août 2026 |
| Français                    | Oui                      |
| Fon natif                   | Non documenté            |
| Licence du code             | MIT                      |
| Licence des poids originaux | CC-BY-4.0                |

#### **Pourquoi Pocket TTS est stratégique**

Contrairement à un gros modèle de synthèse vocale nécessitant un GPU permanent, Pocket TTS permettrait de servir Kanbior sur une infrastructure CPU relativement économique.

L'existence d'Africa TTS montre également que cette architecture peut être adaptée à un grand nombre de langues africaines.

Les principales difficultés seront l'adaptation du tokenizer, la représentation des tons et l'apprentissage des phonèmes propres au Fon.

Sources : [dépôt officiel Pocket TTS](https://github.com/kyutai-labs/pocket-tts?utm_source=chatgpt.com), [documentation](https://kyutai-labs.github.io/pocket-tts/?utm_source=chatgpt.com).

**Verdict : probablement le meilleur candidat à expérimenter si l'objectif est une voix naturelle, conversationnelle et utilisable en streaming sur CPU.**

### **4.3. Matcha-TTS — une alternative moderne à VITS**

Matcha-TTS utilise une architecture de génération acoustique basée sur le conditional flow matching, associée à un vocodeur comme HiFi-GAN.

Texte / phonèmes Fon

**Matcha-TTS**

Génération du spectrogramme Mel

**HiFi-GAN**

Vocodeur

Signal audio

Il permet de dissocier l'apprentissage linguistique de la génération acoustique.

Cela ouvre une troisième stratégie intéressante : entraîner la partie linguistique sur le Fon tout en réutilisant un vocodeur préentraîné sous licence permissive.

Des versions optimisées pour des appareils comme le Raspberry Pi existent, mais le vocodeur peut devenir le principal facteur limitant de l'inférence.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



**Verdict : très bonne alternative technique si Piper et Pocket TTS ne donnent pas satisfaction.**

### **4.4. Qwen3-TTS 0.6B — le candidat orienté qualité**

Qwen3-TTS représente une génération plus récente de modèles vocaux fondés sur des architectures autorégressives et des tokens audio.

| Caractéristique       | Valeur                   |
| --------------------- | ------------------------ |
| Modèle                | Qwen3-TTS-12Hz-0.6B-Base |
| Paramètres            | 600 M                    |
| Licence               | Apache 2.0               |
| Langues officielles   | 10                       |
| Fon                   | Non                      |
| Streaming             | Oui                      |
| Fine-tuning           | Documenté                |
| Inférence privilégiée | GPU                      |

Sur les benchmarks publiés par ses auteurs, le modèle 0.6B obtient notamment un WER de 1,32 % en anglais sur Seed-TTS. Ces résultats indiquent un bon potentiel de fidélité textuelle, mais ne permettent pas de prédire ses performances en Fon.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



Son tokenizer de texte et son codec audio peuvent être intéressants pour l'apprentissage d'une nouvelle langue. Cependant, le fine-tuning documenté pour adapter une voix n'est pas une preuve que l'ajout d'une langue entièrement nouvelle fonctionnera sans modifications.

**Verdict : à conserver comme candidat orienté qualité, pour déterminer si un modèle plus lourd apporte un gain réellement significatif par rapport à Piper et Pocket TTS.**

Sources : [code officiel](https://github.com/QwenLM/Qwen3-TTS?utm_source=chatgpt.com), [checkpoint 0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base?utm_source=chatgpt.com).

### **4.5. Les autres candidats**

| Modèle     | Intérêt                                          | Limite principale                                                 |
| ---------- | ------------------------------------------------ | ----------------------------------------------------------------- |
| Kokoro-82M | Compact, naturel, permissif                      | Pipeline d'ajout d'une nouvelle langue moins clairement documenté |
| SpeechT5   | Fine-tuning accessible, licence MIT              | Préentraînement anglais et génération autorégressive              |
| MeloTTS    | Multilingue, CPU, taille réduite                 | Adaptation du frontend linguistique et des composants BERT        |
| ToucanTTS  | Représentation articulatoire, langues peu dotées | Complexité du pipeline, licence de chaque composant à auditer     |
| MagpieTTS  | Multilingue, représentation IPA                  | Plus lourd et davantage orienté GPU                               |
| CosyVoice  | Streaming et qualité élevée                      | Coût d'inférence et adaptation au Fon incertains                  |

ToucanTTS mérite une attention particulière en raison de son utilisation de représentations articulatoires et de son objectif de couverture de plusieurs milliers de langues. Toutefois, l'annonce d'une couverture linguistique ne garantit ni la présence de données d'entraînement en Fon ni une prononciation fiable.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://toucantts.com\&sz=32)

toucantts.com+3



## **5. Les modèles à exclure pour des raisons commerciales**

Plusieurs modèles très performants pourraient sembler adaptés à Kanbior, mais leurs licences empêchent de les retenir comme base d'entraînement commercial.

| Modèle     | Licence du code               | Licence des poids          | Décision |
| ---------- | ----------------------------- | -------------------------- | -------- |
| MMS-TTS    | MIT                           | CC-BY-NC                   | Exclure  |
| F5-TTS     | MIT                           | CC-BY-NC                   | Exclure  |
| XTTS-v2    | MPL 2.0 pour le fork maintenu | Coqui Public Model License | Exclure  |
| OmniVoice  | Apache 2.0                    | CC-BY-NC                   | Exclure  |
| Africa TTS | MIT                           | CC-BY-NC 4.0               | Exclure  |
| Afri-VITS  | À vérifier                    | CC-BY-NC 4.0               | Exclure  |

[image](https://www.google.com/s2/favicons?domain=https://www.assemblyai.com\&sz=32)

AssemblyAI+4

Le fine-tuning, la quantification ou la conversion ONNX d'un modèle non commercial ne suppriment pas les restrictions des poids d'origine.

En revanche, il est possible de réutiliser une architecture dont le code est permissif et de l'entraîner indépendamment sur des données autorisées.

## **6. Exploitation des données disponibles**

### **6.1. JML-fon-speech-v1**

Kanbior bénéficie déjà d'un volume de données particulièrement intéressant pour une langue peu dotée.

Audio transcrit

# **64 600**

Enregistrements environ

Locuteurs transcrits

# **650**

Données utilisées pour l'ASR Kanbior

# **\~105 h**

Fréquence d'échantillonnage

# **16 kHz**

Le corpus est sous licence CC-BY-SA-4.0, permettant en principe une exploitation commerciale sous réserve du respect de ses obligations.&#x20;

Kanbior state of the art for Fon speech AI.pdf



Cependant, il existe une différence fondamentale entre l'entraînement ASR et l'entraînement TTS.

| Critère                     | ASR                              | TTS                    |
| --------------------------- | -------------------------------- | ---------------------- |
| Diversité des locuteurs     | Très importante                  | Utile si maîtrisée     |
| Bruits ambiants             | Favorisent parfois la robustesse | Indésirables           |
| Qualité du microphone       | Secondaire                       | Très importante        |
| Variabilité acoustique      | Avantage                         | Potentiel inconvénient |
| Transcription approximative | Parfois tolérable                | Problématique          |
| Cohérence des tons          | Importante                       | Fondamentale           |
| Silence et respiration      | Souvent tolérables               | À contrôler            |

**105 heures de données ASR ne représentent donc pas nécessairement 105 heures de données utilisables pour le TTS.**

Un corpus de 15 heures propre, acoustiquement cohérent et correctement annoté peut être préférable à 100 heures d'enregistrements bruités.

### **6.2. WAXAL : une opportunité de transfert interlangue**

Google Research a publié en 2026 le corpus WAXAL, destiné aux technologies vocales pour les langues africaines.

La partie TTS comprend notamment :

- L'éwé (Ewe).
- Le baoulé.
- Le yoruba.
- L'akan.
- Le haoussa.
- L'igbo.

Le corpus TTS publié sur Hugging Face contient plus de 180 heures d'enregistrements de haute qualité. Les données de l'Université du Ghana, dont l'éwé, sont sous licence CC-BY-4.0.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://research.google\&sz=32)

research.google+1



C'est particulièrement intéressant, car l'éwé et le Fon appartiennent tous deux au groupe linguistique Gbe.

Proximité linguistique simplifiée entre l'éwé et le Fon.

Cette proximité offre une piste de recherche supplémentaire :

**Checkpoint commercial générique → adaptation sur l'éwé → fine-tuning sur le Fon.**

L'étape intermédiaire pourrait améliorer la représentation de certains phonèmes et phénomènes tonals communs.

Cette hypothèse doit néanmoins être comparée à un fine-tuning direct sur le Fon pour vérifier qu'elle apporte un bénéfice réel.

Source : [Google WAXAL sur Hugging Face](https://huggingface.co/datasets/google/WaxalNLP?utm_source=chatgpt.com).

### **6.3. Constitution du corpus TTS Kanbior**

Je recommande de constituer trois sous-ensembles à partir de l'audit de JML.

| Dataset                    | Volume cible    | Objectif                 |
| -------------------------- | --------------- | ------------------------ |
| `kanbior-fon-tts-clean`    | 10–20 h         | Première expérimentation |
| `kanbior-fon-tts-extended` | 30–60 h         | Entraînement approfondi  |
| `kanbior-fon-tts-eval`     | 300–500 phrases | Benchmark indépendant    |

Ces volumes sont des objectifs de préparation, pas des volumes propres déjà identifiés dans JML.

Le pipeline de préparation devra inclure la détection des silences, du clipping, du bruit, des erreurs de transcription, des incohérences tonales et des doublons, ainsi qu'une sélection des locuteurs et la vérification de leurs autorisations.

Une collecte complémentaire de 5 à 10 heures auprès d'un locuteur natif, enregistré dans de bonnes conditions et ayant explicitement consenti à la synthèse de sa voix, pourrait également apporter un gain substantiel.

## **7. Particularités linguistiques du Fon**

Le principal risque n'est pas uniquement de produire une voix artificielle. Il est de produire une voix naturelle qui prononce des mots incorrectement et en modifie le sens.

Une étude publiée en juillet 2026 sur l'Efik, autre langue tonale africaine, a comparé plusieurs architectures sur trois heures de données.

MMS-TTS a obtenu le meilleur MOS, avec **3,80 ± 0,63 sur 5**, mais des erreurs tonales subsistaient. Les autres architectures présentaient également des difficultés prosodiques.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org



Cela montre que même un bon score de naturalité ne garantit pas la fidélité linguistique.

### **7.1. Représentation des tons**

Deux stratégies devront être comparées lors de l'entraînement.

**Grapheme-based TTS**

## **Kèmi ná yi**

Caractères Unicode

Accents conservés

Modèle TTS

Plus simple à implémenter.

**Phoneme-based TTS**

## **Kèmi ná yi**

Phonèmes IPA

Représentation explicite des tons

Modèle TTS

Potentiellement plus robuste linguistiquement.

Pour la première version, je privilégierais une approche graphémique avec normalisation Unicode NFC et conservation systématique des diacritiques.

En parallèle, il faudra évaluer le projet [africa-g2p](https://github.com/AfriSpeech/africa-g2p?utm_source=chatgpt.com) pour déterminer s'il fournit une représentation phonétique suffisamment fiable pour le Fon.

Le choix final devra être fondé sur une évaluation par des locuteurs natifs.

### **7.2. Text Normalization**

Le modèle TTS ne devrait pas être chargé de deviner comment prononcer toutes les représentations textuelles possibles.

Un composant dédié devra gérer les nombres, les montants en FCFA, les dates, les heures, les acronymes, les noms propres, les numéros de téléphone et les mots français intégrés dans une phrase en Fon.

Ces transformations doivent intervenir avant la synthèse vocale.

## **8. Entraîner à partir de zéro ou faire du fine-tuning ?**

C'est la décision centrale de cet état de l'art.

Il existe en réalité quatre stratégies possibles.

### **8.1. Entraînement complet à partir de zéro**

Initialiser tous les paramètres de VITS aléatoirement et entraîner le modèle exclusivement avec les données Fon.

VITS — paramètres aléatoires

Entraînement sur corpus Fon

**Kanbior Fon TTS**

Avantages : maîtrise de la provenance des poids, architecture entièrement spécialisée, absence de biais linguistiques directement hérités d'un autre checkpoint.

Inconvénients : convergence potentiellement plus lente, apprentissage acoustique complet, sensibilité à la qualité des données, risques d'instabilité et besoin de davantage d'expérimentations.

Cette approche est viable pour une petite architecture comme VITS. Elle est beaucoup moins intéressante pour un modèle génératif de plusieurs centaines de millions de paramètres.

### **8.2. Fine-tuning direct d'un modèle commercial**

Pocket TTS ou Piper préentraîné

Adaptation tokenizer / phonèmes + fine-tuning Fon

**Kanbior Fon TTS**

Le modèle bénéficie déjà d'une connaissance des propriétés acoustiques de la parole humaine, des alignements, des durées, des transitions entre sons et de certains phénomènes prosodiques.

Même lorsqu'il n'a jamais appris le Fon, ces connaissances peuvent être transférées.

Il faudra toutefois ajouter ou réinitialiser les représentations des nouveaux symboles et permettre au modèle de réellement apprendre une nouvelle langue.

Un simple fine-tuning de quelques couches destiné à adapter une voix risque de ne pas suffire.

### **8.3. Fine-tuning interlangue via l'éwé**

Checkpoint préentraîné commercial

Apprentissage de l'éwé sur WAXAL

Fine-tuning sur JML Fon

**Kanbior Fon TTS**

C'est la piste potentiellement la plus originale scientifiquement.

Elle exploite une langue proche et disposant de données TTS de meilleure qualité acoustique. Elle pourrait permettre de compenser certaines faiblesses de JML, mais elle augmente les coûts d'entraînement et introduit un risque de transfert d'accent ou de caractéristiques propres à l'éwé.

### **8.4. Entraînement partiel avec vocodeur préentraîné**

Cette stratégie intermédiaire consiste à entraîner l'encodeur linguistique et le modèle acoustique sur le Fon, tout en réutilisant un vocodeur préentraîné commercialement exploitable.

Elle permet de spécialiser fortement le modèle dans la langue cible sans devoir réapprendre entièrement la génération du signal audio.

Matcha-TTS avec HiFi-GAN constitue une architecture intéressante pour expérimenter cette approche.

### **8.5. Comparaison des stratégies**

| Critère                     | From scratch    | Fine-tuning direct     | Transfert via éwé     | Vocodeur préentraîné |
| --------------------------- | --------------- | ---------------------- | --------------------- | -------------------- |
| Coût d'entraînement relatif | Élevé           | Faible à moyen         | Moyen à élevé         | Moyen                |
| Données nécessaires         | Plus nombreuses | Moins nombreuses       | Deux corpus           | Intermédiaire        |
| Convergence                 | Plus incertaine | Potentiellement rapide | Progressive           | Intermédiaire        |
| Qualité acoustique initiale | Aucune          | Préapprise             | Préapprise            | Préapprise en partie |
| Spécialisation Fon          | Forte           | Forte après adaptation | Potentiellement forte | Forte                |
| Complexité technique        | Moyenne         | Moyenne                | Élevée                | Moyenne              |
| Priorité                    | 2               | **1**                  | 3                     | 4                    |

**Point essentiel : le fine-tuning n'augmente pas nécessairement le coût d'inférence.**

Un Piper de 15 millions de paramètres entraîné à partir de zéro et un Piper de 15 millions de paramètres fine-tuné auront sensiblement les mêmes besoins en production, à architecture et paramètres d'exécution identiques.

Il faut donc distinguer deux décisions :

- L'architecture détermine principalement les ressources d'inférence.
- La stratégie d'entraînement détermine principalement la convergence, les données nécessaires et la qualité obtenue.

Le fine-tuning d'un petit modèle constitue ainsi un compromis beaucoup plus intéressant que le fine-tuning d'un modèle de plusieurs milliards de paramètres lorsque le CPU est la cible.

## **9. Comparaison des ressources d'inférence**

### **9.1. Taille théorique des poids**

### **Taille des modèles selon la précision numérique**

FP32FP16INT8

Piper Medium

60 Mo

MMS-TTS

144 Mo

Kokoro

328 Mo

Pocket TTS

400 Mo

SpeechT5

576 Mo

MagpieTTS

1.46 Go

Qwen3-TTS 0.6B

2.40 Go

Qwen3-TTS 1.7B

6.80 Go

Estimation théorique du stockage des paramètres, sans vocodeur supplémentaire, activations, cache, runtime ou surcoût de quantification. La compatibilité effective de chaque modèle avec INT8 n'est pas garantie.

Pour Kanbior, il ne faut pas reproduire automatiquement la démarche de quantification INT8 utilisée pour Whisper.

Contrairement à un modèle ASR, le modèle TTS doit générer directement un signal acoustique. Une quantification agressive peut introduire des artefacts, modifier la prosodie ou dégrader certains contrastes tonals.

La quantification doit donc intervenir après validation de la qualité du modèle original.

### **9.2. Critères de performance**

| Indicateur               | Cible proposée               |
| ------------------------ | ---------------------------- |
| Taille du modèle déployé | Moins de 500 Mo              |
| RAM consommée            | Moins de 2 Go                |
| Infrastructure minimale  | 2–4 vCPU                     |
| Real-Time Factor         | Inférieur à 0,5              |
| Time to First Audio      | Moins de 500 ms si streaming |
| GPU obligatoire          | Non                          |
| Traitement local         | Souhaitable                  |

Ces valeurs constituent des objectifs de conception pour Kanbior, et non des performances déjà mesurées.

Le Real-Time Factor (RTF) constitue une métrique essentielle :

\\[ RTF=\frac{\text{Temps de génération}}{\text{Durée audio générée}} \\]

Par exemple, pour générer dix secondes de parole :

**RTF 0,1**

1 s

10× plus rapide que le temps réel

**RTF 0,5**

5 s

2× plus rapide

**RTF 1,0**

10 s

Temps réel

**RTF 1,5**

15 s

Plus lent que le temps réel

Un RTF faible réduit le coût de génération, augmente la capacité à servir plusieurs utilisateurs et améliore la fluidité d'un assistant vocal.

### **9.3. Simulation du coût d'inférence**

### **Estimateur du coût de génération mensuel**

Requêtes mensuelles

#### **100 000**

Durée moyenne d'une réponse

#### **15 s**

Real-Time Factor

#### **0.40**

Coût horaire hypothétique de la machine ($)

Temps machine

# **166,7 h**

Coût de calcul estimé

# **16,67 $**

Simulation avec facturation du temps de calcul actif. Hors périodes d'inactivité, concurrence, réseau, stockage, démarrage et coûts des autres composants Kanbior.

## **10. Comment mesurer la qualité du modèle TTS ?**

Contrairement à l'ASR, où le WER constitue une métrique standard, l'évaluation TTS nécessite plusieurs dimensions.

| Métrique           | Mesure                            | Importance |
| ------------------ | --------------------------------- | ---------- |
| MOS                | Naturalité de la parole           | Très haute |
| WER                | Intelligibilité des mots          | Très haute |
| CER                | Fidélité des caractères           | Haute      |
| Tone Error Rate    | Erreurs de tons lexicaux          | Critique   |
| Évaluation humaine | Fidélité du sens et prononciation | Critique   |
| RTF                | Vitesse d'inférence               | Très haute |
| TTFA               | Latence du premier audio          | Haute      |
| Peak RAM           | Consommation mémoire              | Haute      |
| Model Size         | Taille de déploiement             | Haute      |

### **10.1. Réutilisation de kanbior-fon-asr-v1**

Kanbior dispose déjà d'un avantage stratégique : son propre modèle ASR Fon.

Il pourra être utilisé pour automatiser une partie du benchmark TTS.

Ton modèle ASR ayant obtenu un WER de 20,5 % et un CER de 8,5 % sur des locuteurs inconnus, il est intéressant pour comparer automatiquement les différentes versions du TTS.

Cependant, le WER mesuré inclura aussi les erreurs de l'ASR lui-même. Il faudra mesurer ses performances sur les enregistrements humains correspondant aux mêmes textes et ne pas considérer ses retranscriptions comme une vérité absolue.

Il faudra également éviter les fuites de données entre les corpus d'entraînement et d'évaluation.

### **10.2. Benchmark humain**

Je recommande un ensemble de 300 à 500 phrases jamais vues pendant l'entraînement, couvrant notamment les conversations quotidiennes, les montants, les dates, les phrases longues, les questions, le mélange Fon-français et les paires minimales différenciées par leurs tons.

Une évaluation en aveugle par 10 à 20 locuteurs natifs permettrait de comparer les différents modèles sur la naturalité, l'intelligibilité et la fidélité sémantique.

Pour un modèle mono-locuteur, le test doit contenir des phrases nouvelles, mais pas nécessairement un locuteur inconnu comme on l'exigerait lors d'un benchmark ASR.

## **11. Classement multicritère pour Kanbior**

Le classement suivant est une estimation de la pertinence technique des candidats, et non un benchmark de leurs performances réelles en Fon.

J'ai volontairement donné un poids important à l'inférence, car Kanbior intégrera également un ASR, un modèle de traduction et un LLM.

### **Classement ajustable des architectures**

Modifier l'importance relative de chaque critère.

Potentiel de qualité

35 %

Efficacité d'inférence

35 %

Facilité d'adaptation

20 %

Facilité de déploiement

10 %

1\.

**Piper / VITS**

8.93/10

2\.

Pocket TTS

8.80/10

3\.

Matcha-TTS

8.00/10

4\.

Kokoro

7.72/10

5\.

MeloTTS

7.60/10

6\.

SpeechT5

6.85/10

7\.

Qwen3-TTS 0.6B

6.80/10

8\.

MagpieTTS

6.53/10

Réinitialiser la pondérationScores qualitatifs attribués pour prioriser les expérimentations. Ils ne représentent ni des MOS mesurés ni des performances certifiées en Fon. Les modèles non commerciaux ont été écartés au préalable.

## **12. Plan d'expérimentation recommandé**

Il serait inutile d'entraîner dix modèles différents. Je recommande de concentrer les ressources initiales sur trois expériences et une baseline.

### **E0 — Baseline MMS-TTS-Fon**

Sans entraînement

Évaluer le modèle Fon existant dans un cadre de recherche compatible avec sa licence.

### **E1 — Fine-tuning Piper Medium**

Checkpoint CC0 · 10–20 h Fon

Obtenir le modèle de référence commercial le plus léger possible.

### **E2 — Fine-tuning Pocket TTS**

Checkpoint original CC-BY-4.0 · 10–20 h Fon

Évaluer le gain de naturalité, les capacités conversationnelles et le streaming CPU.

### **E3 — Piper / VITS from scratch**

Même architecture et même corpus que E1

Mesurer objectivement le bénéfice du préentraînement.

L'expérience E3 est importante : sans elle, il serait impossible de conclure scientifiquement que le fine-tuning est préférable à un entraînement à partir de zéro pour le Fon.

### **Budget initial indicatif**

| Expérience            | GPU      | Budget GPU-heure prévisionnel |
| --------------------- | -------- | ----------------------------- |
| Baseline MMS          | Aucun    | 0 h                           |
| Piper fine-tuning     | 16–24 Go | 20–60 h                       |
| Pocket TTS adaptation | 24–48 Go | 40–120 h                      |
| Piper from scratch    | 16–24 Go | 60–150 h                      |
| **Total**             |          | **120–330 h**                 |

Ces valeurs constituent des enveloppes de planification, avec arrêt anticipé des expériences en cas de mauvaise convergence. Elles ne sont pas issues d'entraînements effectifs sur JML.

### **Processus de sélection final**

## **13. Recommandation d'architecture pour la production**

L'objectif pourrait être de proposer, à terme, deux variantes du modèle Kanbior, si les expérimentations justifient cette distinction.

**kanbior-fon-tts-lite**

# **\~15 M**

Paramètres

Base : Piper / VITS

Runtime : ONNX

CPU et mobile

Faible latence

Infrastructure minimale

**kanbior-fon-tts-standard**

# **\~100 M**

Paramètres

Base : Pocket TTS

Runtime : PyTorch / ONNX si validé

CPU

Streaming audio

Naturalité privilégiée

Cette approche ressemble à la logique des familles Whisper Tiny, Base, Small et Medium : permettre à l'utilisateur de choisir le modèle en fonction de ses ressources.

Il ne serait toutefois pas nécessaire de maintenir deux variantes dès la V1. Le meilleur modèle issu du benchmark doit être livré en premier.

## **14. Licence et commercialisation**

Il faudra constituer un registre de provenance précis avant de publier le modèle.

| Composant              | Licence envisagée        |
| ---------------------- | ------------------------ |
| Piper Thorsten Scratch | CC0                      |
| Pocket TTS original    | CC-BY-4.0                |
| JML Fon Speech         | CC-BY-SA-4.0             |
| WAXAL Ewe TTS          | CC-BY-4.0                |
| Piper historique       | MIT                      |
| Piper maintenu         | GPL-3.0                  |
| ONNX Runtime           | MIT                      |
| Modèle Kanbior final   | À déterminer après audit |

Trois points devront être vérifiés : les droits sur les poids préentraînés et leurs dépendances, les obligations ShareAlike de JML et les autorisations liées à l'exploitation de la voix des locuteurs.

Une licence autorisant l'utilisation commerciale d'enregistrements ne constitue pas automatiquement un consentement au clonage vocal individuel. De même, commercialiser uniquement un service ne garantit pas à lui seul que toutes les obligations juridiques sont satisfaites.

Une validation auprès de J'aime Ma Langue, de l'ASIN ou d'un spécialiste de la propriété intellectuelle serait pertinente avant la commercialisation.

## **15. Conclusion générale de l'état de l'art**

### **Quel modèle choisir ?**

| Objectif                                | Modèle recommandé       |
| --------------------------------------- | ----------------------- |
| Meilleur compromis potentiel global     | **Pocket TTS**          |
| Meilleure efficacité d'inférence        | **Piper / VITS Medium** |
| Baseline scientifique Fon existante     | MMS-TTS-Fon             |
| Alternative acoustique moderne          | Matcha-TTS              |
| Expérimentation haut de gamme           | Qwen3-TTS 0.6B          |
| Recherche sur le transfert linguistique | WAXAL Éwé → Fon         |

### **Faut-il entraîner à partir de zéro ?**

**Je recommande de privilégier le fine-tuning d'un modèle compact et commercialement exploitable, plutôt que l'entraînement intégral à partir de zéro.**

Cette recommandation repose sur cinq raisons :

1. Un modèle préentraîné possède déjà des connaissances acoustiques transférables.
2. La qualité des données JML n'a pas été spécifiquement optimisée pour la synthèse vocale.
3. Le fine-tuning peut réduire le temps d'entraînement et accélérer les expérimentations.
4. Le préentraînement n'impose pas de surcoût d'inférence à architecture identique.
5. Des checkpoints compacts et commercialement exploitables existent désormais, ce qui remet en question la recommandation initiale de Kanbior.

L'entraînement from scratch reste pertinent comme expérience comparative ou comme solution de repli si le transfert linguistique s'avère inefficace.

### **Ma recommandation pour démarrer**

## **Première expérimentation**

Architecture

Piper / VITS Medium

Checkpoint

Thorsten-Voice/Piper-Scratch

Licence de base

CC0

Entraînement

Full fine-tuning interlangue

Dataset

JML Fon — 10 à 20 h nettoyées

Taille cible

\~60 Mo en FP32

Déploiement

ONNX Runtime sur CPU

Baseline

MMS-TTS-Fon

Challenger

Pocket TTS 100M

Consulter le checkpoint Piper Consulter Pocket TTS&#x20;

## **16. Principales références scientifiques et techniques**

| Référence                                                                                               | Contribution                                    |
| ------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| [VITS — Kim et al., 2021](https://arxiv.org/abs/2106.06103?utm_source=chatgpt.com)                      | Architecture TTS end-to-end                     |
| [Massively Multilingual Speech — Meta, 2023](https://arxiv.org/abs/2305.13516?utm_source=chatgpt.com)   | Synthèse vocale multilingue incluant le Fon     |
| [Matcha-TTS — 2023](https://arxiv.org/abs/2309.03199?utm_source=chatgpt.com)                            | Flow matching pour TTS efficace                 |
| [SpeechT5](https://arxiv.org/abs/2110.07205?utm_source=chatgpt.com)                                     | Préentraînement parole et texte                 |
| [Pocket TTS — Kyutai, 2026](https://kyutai-labs.github.io/pocket-tts/?utm_source=chatgpt.com)           | Synthèse vocale compacte sur CPU                |
| [WAXAL — Google Research, 2026](https://huggingface.co/datasets/google/WaxalNLP?utm_source=chatgpt.com) | Corpus TTS africains commercialisables          |
| [TTS for Efik — 2026](https://arxiv.org/abs/2607.04515?utm_source=chatgpt.com)                          | Comparaison TTS sur une langue tonale africaine |
| [Africa TTS — AfriSpeech, 2026](https://github.com/AfriSpeech/africa-tts?utm_source=chatgpt.com)        | Adaptation de Pocket TTS aux langues africaines |
| [Piper Training Guide](https://github.com/rhasspy/piper/blob/master/TRAINING.md?utm_source=chatgpt.com) | Fine-tuning interlangue et export ONNX          |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS?utm_source=chatgpt.com)                                 | Modèles génératifs vocaux récents               |

### **Limites de cet état de l'art**

Aucun benchmark commun vérifié ne permet actuellement de classer Piper, Pocket TTS et Qwen3-TTS selon leur qualité effective après adaptation au Fon. Les classements, budgets GPU et objectifs d'inférence proposés constituent donc des hypothèses de recherche. Certains composants et checkpoints devront aussi faire l'objet d'un audit de licence complémentaire.

Je n'ai effectué ni entraînement ni test audio réel dans le cadre de cette recherche.

**L'opportunité scientifique et industrielle pour Kanbior est précisément de produire ce benchmark : démontrer qu'un modèle TTS spécialisé en Fon, de 15 à 100 millions de paramètres, peut offrir une parole naturelle, linguistiquement correcte et commercialisable, sans nécessiter de GPU en production.**
-----

The data consiste of transcribed and untranscribed part, is there any what to also use the unstranscribed ones, be his test, be short , and justify your choices

---
Yes. The approximately **95,700 untranscribed recordings** in JML can also contribute to training `kanbior-fon-tts-v1`.

There are three main approaches:

| Approach                        | Method                                                                                                                        | Recommendation                          |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| **Pseudo-labeling**             | Transcribe audio using `kanbior-fon-asr-v1`, filter unreliable predictions, then train TTS on the resulting audio-text pairs. | Highly recommended                      |
| **Vocoder pretraining**         | Train or adapt a vocoder such as HiFi-GAN using audio alone.                                                                  | Recommended for acoustic quality        |
| **Self-supervised pretraining** | Learn acoustic representations from untranscribed speech before supervised TTS training.                                      | Possible, but computationally expensive |

### **My recommended strategy**

1. Fine-tune the TTS model using clean, human-transcribed recordings.
2. Automatically transcribe the remaining recordings using Kanbior ASR.
3. Select recordings with high transcription confidence, good acoustic quality and reliable tonal annotations.
4. Continue training the TTS model using both datasets.

**Why pseudo-labeling?** You already have a Fon ASR model with a WER of 20.5%, making this approach relatively inexpensive without increasing TTS inference costs.

However, that WER is still significant. Incorrect pseudo-labels could teach the TTS model incorrect pronunciations and tones. Confidence filtering and human validation of a representative sample are therefore essential.

**Conclusion:** prioritize supervised fine-tuning followed by semi-supervised learning through pseudo-labeling. Use untranscribed audio for vocoder adaptation if necessary. Avoid expensive self-supervised pretraining for the V1.

This recommendation remains to be validated experimentally on JML.
----

Is that the good order or first use pseudo labels ?

-----

I would actually recommend **using pseudo-labeled data first, then fine-tuning on human-transcribed data**.

### **Recommended training order**

| Stage                       | Dataset                                               | Objective                                                 |
| --------------------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| 1. Pretrained model         | Piper or Pocket TTS                                   | Leverage existing acoustic knowledge                      |
| 2. Intermediate fine-tuning | Filtered pseudo-labeled data + human-transcribed data | Learn Fon phonetics, pronunciation and acoustic diversity |
| 3. Final fine-tuning        | High-quality human-transcribed data only              | Refine pronunciation, tonal accuracy and naturalness      |
| 4. Evaluation               | Independent human-validated test set                  | Measure final quality                                     |

### **Why this order?**

- **Learn broadly first:** exploit the larger dataset to acquire Fon-specific acoustic and linguistic knowledge.
- **Refine with reliable data:** correct potential biases introduced by inaccurate pseudo-labels.
- **Avoid final-stage degradation:** finishing with noisy pseudo-labels could undermine previously learned correct pronunciations.

Given that Fon is a tonal language and your ASR has a WER of 20.5%, I would not train exclusively on pseudo-labeled data during stage 2.

**My preferred strategy:**

Pretrained model → Mixed-quality data → High-quality data → Evaluation.

This is a strong starting hypothesis, but it should be compared against fine-tuning exclusively on human-transcribed data to verify that pseudo-labeling actually improves TTS quality.

---- 
Son you think the only transcribed is enough, do you have an idea of the size of datât use by Facebook one ?
----

Yes. I think the human-transcribed portion of JML is sufficient to develop a good first Fon TTS model, provided enough recordings are acoustically clean.

With approximately **105 hours of transcribed speech**, you already have a substantial dataset for fine-tuning a compact TTS model. The main uncertainty is audio quality and the distribution of recordings across speakers, rather than the total volume.

### **How much data did Meta use?**

The official [MMS-TTS-Fon model card](https://huggingface.co/facebook/mms-tts-fon?utm_source=chatgpt.com) confirms that Meta trained a separate VITS model for each language. However, it does not specify the exact number of training hours for Fon.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



| Characteristic | Meta MMS-TTS-Fon | Kanbior TTS (proposed)             |
| -------------- | ---------------- | ---------------------------------- |
| Architecture   | VITS             | Piper / Pocket TTS                 |
| Parameters     | 36.3 M           | \~15 M / 100 M                     |
| Training data  | Bible readings   | JML everyday sentences             |
| Training hours | Not verified     | \~105 h available before filtering |
| Commercial use | No               | Intended                           |

### **How much data is generally needed for TTS?**

| Training scenario                    | Indicative data requirements |
| ------------------------------------ | ---------------------------- |
| Single-speaker fine-tuning           | 1–5 h                        |
| Single-speaker training from scratch | 10–25 h                      |
| Cross-language fine-tuning           | 10–30 h                      |
| Multi-speaker TTS                    | 50–100+ h                    |

These are practical guidelines, not guaranteed minimums.

**My recommendation:** even if you discard 70% of your transcribed dataset during quality filtering, you would still have around **31.5 hours** of speech.

That is a reasonable starting point for cross-language fine-tuning. I would prioritize data quality over pseudo-labeling for V1, especially because Fon is a tonal language.

The main challenge is not necessarily having enough data, but having enough **clean, correctly transcribed and acoustically consistent speech per speaker**.