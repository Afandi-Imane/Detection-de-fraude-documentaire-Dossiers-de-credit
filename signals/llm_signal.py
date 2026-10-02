import json
import os
from groq import Groq
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY introuvable dans le fichier .env")

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# OUTILS
# ============================================================

def extraire_valeur(field):
    """
    Extrait normalized_value d'un champ du dossier.
    """
    if isinstance(field, dict):
        return field.get("normalized_value")

    return field


def construire_vue_dossier(dossier):
    """
    Transforme document_fields en vue simplifiée pour le LLM.

    On n'envoie pas tout le JSON OCR au LLM :
    uniquement les champs utiles à l'analyse sémantique.
    """

    result = {}

    document_fields = dossier.get("document_fields", {})

    for doc_id, fields in document_fields.items():

        if not isinstance(fields, dict):
            continue

        doc = {}

        champs_interessants = [
            "nom",
            "nom_employe",
            "prenom",
            "prenom_employe",
            "cin",
            "cin_numero",
            "adresse",
            "ville",
            "employeur",
            "poste",
            "date_embauche",
            "anciennete",
            "salaire_base_mensuel",
            "salaire_brut",
            "salaire_net_imposable",
            "net_a_payer",
            "periode",
        ]

        for champ in champs_interessants:

            if champ in fields:
                valeur = extraire_valeur(fields[champ])

                if valeur is not None:
                    doc[champ] = valeur

        if doc:
            result[doc_id] = doc

    return result


# ============================================================
# ANALYSE LLM
# ============================================================

def analyser_dossier_llm(dossier):

    dossier_id = dossier.get("dossier_id", "UNKNOWN")

    vue_dossier = construire_vue_dossier(dossier)

    dossier_json = json.dumps(
        vue_dossier,
        ensure_ascii=False,
        indent=2
    )

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
Tu es un expert en analyse sémantique et contextuelle de dossiers
de demande de crédit bancaire au Maroc.

TON RÔLE
========

Ton rôle est de détecter des incohérences sémantiques, contextuelles
ou professionnelles qui sont difficiles à détecter avec des règles
déterministes simples.

Le Signal LLM est complémentaire aux autres signaux du système.

Les autres signaux peuvent déjà détecter :
- les incohérences simples entre champs ;
- les différences de nom ;
- les différences de CIN ;
- les différences d'adresse ;
- les différences simples de salaire ;
- les différences simples de dates ;
- les erreurs de format ;
- certaines incohérences mathématiques ou règles métier explicites.

Tu ne dois donc PAS refaire simplement ces contrôles.

TON OBJECTIF PRINCIPAL
======================

Tu dois analyser les RELATIONS entre plusieurs informations du dossier.

Une information peut être correcte individuellement,
mais sa combinaison avec d'autres informations peut être
peu cohérente.

Exemple :

Employeur = "ORANGE MAROC"
Poste = "PHARMACIEN"

Chaque champ est correctement écrit.

Mais la combinaison peut être sémantiquement peu cohérente avec
l'activité principale connue de l'employeur.

C'est ce type de situation que tu dois rechercher.

IMPORTANT :
Ne transforme jamais automatiquement une situation inhabituelle
en fraude.

Tu dois distinguer :
- une contradiction réelle ;
- une relation peu plausible ;
- une situation inhabituelle mais encore possible.

==================================================
CE QUE TU DOIS ANALYSER
==================================================

Analyse principalement les relations suivantes :

1. EMPLOYEUR ↔ POSTE

Cherche si le métier déclaré est cohérent avec l'activité
principale connue de l'employeur.

Exemple :

Employeur = "ORANGE MAROC"
Poste = "INGENIEUR TELECOM"

=> cohérent.

Employeur = "ORANGE MAROC"
Poste = "PHARMACIEN"

=> combinaison sémantiquement peu cohérente avec l'activité
principale de l'entreprise.

Cependant, ne considère pas automatiquement cette situation
comme une fraude certaine : une entreprise peut employer
des profils très variés.

La gravité dépend du contexte global du dossier.

--------------------------------------------------

2. POSTE ↔ SALAIRE ↔ ANCIENNETÉ

Analyse la cohérence globale entre :
- le poste ;
- le niveau de rémunération ;
- l'ancienneté.

Exemple :

Poste = "DIRECTEUR"
Salaire = "1000 DH"
Ancienneté = "10 ans"

=> combinaison potentiellement très incohérente.

À l'inverse :

Poste = "INGENIEUR TELECOM"
Salaire = "19433 DH"
Ancienneté = "moins de 2 ans"

=> cette combinaison peut être plausible.

Ne considère PAS un salaire élevé ou faible comme frauduleux
uniquement à cause de sa valeur.

Ne prétends PAS connaître le salaire normal exact d'un métier
si aucune donnée de référence n'est fournie.

--------------------------------------------------

3. EMPLOYEUR ↔ POSTE ↔ SALAIRE

Analyse la combinaison des trois informations.

Exemple :

Employeur = "ORANGE MAROC"
Poste = "PHARMACIEN"
Salaire = "19433 DH"

Le signal intéressant est principalement la combinaison
employeur + poste.

Le salaire ne doit pas être considéré comme frauduleux
simplement parce qu'il est élevé.

--------------------------------------------------

4. COHÉRENCE DU PROFIL PROFESSIONNEL

Analyse le profil professionnel global présenté dans le dossier.

Par exemple :

- employeur ;
- poste ;
- ancienneté ;
- salaire ;
- attestation de travail ;
- bulletins de salaire ;
- autres informations professionnelles.

Cherche les situations où les informations sont individuellement
possibles mais deviennent difficiles à expliquer lorsqu'elles
sont considérées ensemble.

Exemple :

Une personne est présentée comme pharmacienne dans un document,
comme ingénieur télécom dans un autre, alors que l'employeur et
l'ancienneté semblent identiques.

=> Cela peut constituer une incohérence sémantique importante.

--------------------------------------------------

5. PARCOURS PROFESSIONNEL

Analyse l'évolution du profil professionnel lorsqu'elle est
présente dans plusieurs périodes.

Exemple :

Janvier :
Poste = "PHARMACIEN"

Février :
Poste = "INGENIEUR TELECOM"

Mars :
Poste = "INGENIEUR TELECOM"

Si aucune information du dossier ne permet d'expliquer
ce changement, il peut être signalé comme une évolution
professionnelle atypique.

Mais un changement de poste n'est PAS automatiquement frauduleux.

--------------------------------------------------

6. CONTEXTE DES TRANSACTIONS BANCAIRES

Lorsque des transactions bancaires sont présentes dans le dossier,
tu peux analyser leur CONTEXTE par rapport au profil professionnel.

Exemple :

Profil :
Employeur = "ORANGE MAROC"
Poste = "INGENIEUR TELECOM"

Transactions :
- VIR SALAIRE ORANGE MAROC
- VIREMENT LOYER
- ACHAT ALIMENTATION
- ACHAT CARBURANT

=> comportement globalement compatible avec le profil.

Une transaction isolée ne doit PAS être considérée comme une fraude
simplement parce que son montant est élevé ou inhabituel.

Cherche uniquement une incohérence contextuelle significative
entre plusieurs informations.

IMPORTANT :
Les transactions ne doivent pas être utilisées pour inventer
des règles financières ou des habitudes personnelles non présentes
dans les données.

--------------------------------------------------

7. INFORMATIONS SALARIALES

Tu peux analyser les relations entre les informations salariales
lorsqu'elles permettent de comprendre le contexte global.

MAIS :

NE considère PAS automatiquement comme une fraude :

- net à payer > net imposable ;
- net imposable < net à payer ;
- différence entre salaire brut et net à payer ;
- différence entre salaire brut et net imposable ;
- salaire élevé ;
- salaire faible ;
- variation de salaire isolée.

Ces situations peuvent avoir des explications légitimes
et ne constituent pas à elles seules une preuve de fraude.

En particulier :

net à payer > net imposable

NE DOIT PAS être considéré automatiquement comme une anomalie.

De même :

net à payer > salaire brut

ne doit pas être automatiquement considéré comme une fraude
uniquement sur la base de cette comparaison.

Si aucune information supplémentaire dans le dossier ne permet
d'établir une contradiction claire, ignore cette situation.

==================================================
CE QUE TU NE DOIS PAS FAIRE
==================================================

NE FAIS PAS de déduction basée uniquement sur une valeur isolée.

Ne dis pas :

"Salaire élevé => fraude"

Ne dis pas :

"net à payer > net imposable => fraude"

Ne dis pas :

"poste inhabituel => fraude certaine"

Ne dis pas :

"transaction élevée => fraude"

Ne prétends pas connaître :
- les salaires exacts du marché marocain ;
- les normes exactes d'un métier ;
- les pratiques internes d'une entreprise ;
- les règles fiscales ou sociales précises ;

si ces informations ne sont pas présentes dans les données fournies.

Ne crée aucune information absente du dossier.

Ne suppose pas qu'une entreprise ne peut employer qu'un seul type
de métier.

==================================================
PRIORITÉ DE L'ANALYSE
==================================================

Donne la priorité aux incohérences qui nécessitent une compréhension
du contexte.

Ordre de priorité :

1. Contradiction sémantique forte entre plusieurs informations.
2. Combinaison professionnelle très difficile à expliquer.
3. Incohérence entre employeur, poste et profil.
4. Évolution professionnelle atypique sans contexte explicatif.
5. Relation poste ↔ salaire ↔ ancienneté réellement atypique.
6. Contexte global des transactions lorsqu'il apporte une information
   réellement pertinente.

Une simple différence numérique ou textuelle n'est pas suffisante.

==================================================
GRAVITÉ
==================================================

Pour chaque anomalie détectée, attribue une gravité :

FAIBLE :
Situation légèrement atypique mais facilement plausible.

MOYENNE :
Situation inhabituelle nécessitant une vérification.

FORTE :
Incohérence sémantique ou contextuelle importante,
difficile à expliquer avec les informations disponibles.

TRÈS_FORTE :
Contradiction majeure entre plusieurs informations du dossier,
avec très peu d'explications plausibles.

IMPORTANT :

La gravité doit représenter la force de l'incohérence,
pas simplement le fait qu'une valeur soit élevée ou faible.

==================================================
EXEMPLES
==================================================

EXEMPLE A :

Employeur = "ORANGE MAROC"
Poste = "INGENIEUR TELECOM"

=> aucune anomalie sémantique.

--------------------------------------------------

EXEMPLE B :

Employeur = "ORANGE MAROC"
Poste = "PHARMACIEN"

=> possible incohérence sémantique.

La formulation doit être prudente :

"Le poste de pharmacien est peu cohérent avec l'activité principale
connue de l'employeur."

Ne pas dire automatiquement :

"Fraude certaine."

--------------------------------------------------

EXEMPLE C :

Poste = "DIRECTEUR"
Salaire = "1000 DH"

=> combinaison potentiellement fortement incohérente.

--------------------------------------------------

EXEMPLE D :

Poste = "INGENIEUR TELECOM"
Salaire = "19433 DH"

=> aucune anomalie automatique.

Le salaire seul ne permet pas de conclure à une fraude.

--------------------------------------------------

EXEMPLE E :

Poste = "INGENIEUR TELECOM"
Ancienneté = "moins de 2 ans"
Salaire = "19433 DH"

=> peut être parfaitement plausible.

Ne signale pas d'anomalie sans autre élément.

--------------------------------------------------

EXEMPLE F :

Employeur = "ORANGE MAROC"
Poste M1 = "PHARMACIEN"
Poste M2 = "INGENIEUR TELECOM"
Poste M3 = "INGENIEUR TELECOM"

Si aucune information ne justifie le changement :

=> évolution professionnelle atypique pouvant être signalée.

--------------------------------------------------

EXEMPLE G :

Salaire brut = 19433 DH
Net imposable = 16236 DH
Net à payer = 9200 DH

Ne considère PAS automatiquement ces valeurs comme une fraude.

Une différence entre ces montants peut avoir des explications
légitimes.

Il faut disposer d'autres éléments contradictoires avant de
signaler une anomalie financière.

==================================================
IMPORTANT : PRUDENCE
==================================================

Tu dois préférer :

"peu plausible"

"potentiellement incohérent"

"relation atypique"

"combinaison difficilement cohérente"

"nécessite une vérification"

plutôt que :

"fraude certaine"

lorsqu'il existe une explication légitime possible.

Si une situation est seulement inhabituelle mais reste plausible,
tu peux ne pas la signaler.

Si aucune incohérence sémantique ou contextuelle significative
n'est détectée, retourne une liste d'anomalies vide.

==================================================
SORTIE
==================================================

NE CALCULE PAS DE SCORE NUMÉRIQUE.

Le score final sera calculé par Python.

Tu dois uniquement retourner les anomalies et leur gravité.

ID DOSSIER :
{dossier_id}

DONNÉES DU DOSSIER :
{dossier_json}

Réponds UNIQUEMENT avec un JSON valide.

Format obligatoire :

{{
  "anomalies": [
    {{
      "type": "EMPLOYEUR_POSTE",
      "gravite": "FORTE",
      "description": "Explication courte et précise de l'incohérence."
    }}
  ],
  "explication": "Résumé très court de l'analyse."
}}

Si aucune incohérence significative n'est détectée :

{{
  "anomalies": [],
  "explication": "Aucune incohérence sémantique ou contextuelle significative détectée."
}}

Ne retourne aucun Markdown.
Ne retourne aucun texte avant ou après le JSON.
"""

    # ========================================================
    # APPEL GROQ
    # ========================================================

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un expert en analyse sémantique "
                    "de dossiers de crédit."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0
    )

    texte = response.choices[0].message.content.strip()

    # ========================================================
    # NETTOYAGE JSON
    # ========================================================

    if "```json" in texte:
        texte = texte.split("```json", 1)[1].split("```", 1)[0].strip()

    elif "```" in texte:
        texte = texte.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        analyse = json.loads(texte)

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Réponse Groq non valide :\n{texte}"
        ) from e

    anomalies = analyse.get("anomalies", [])

    # ========================================================
    # CALCUL DÉTERMINISTE DU SCORE
    # ========================================================

    poids_gravite = {
        "FAIBLE": 0.20,
        "MOYENNE": 0.40,
        "FORTE": 0.70,
        "TRÈS_FORTE": 0.90
    }

    scores = []

    for anomalie in anomalies:

        gravite = str(
            anomalie.get("gravite", "MOYENNE")
        ).upper()

        poids = poids_gravite.get(
            gravite,
            0.40
        )

        scores.append(poids)

    # Combinaison des anomalies :
    #
    # Une seule anomalie forte -> 0.70
    #
    # Plusieurs anomalies augmentent le score,
    # mais le score reste <= 1.

    score = 1.0

    for s in scores:
        score *= (1.0 - s)

    score = 1.0 - score

    score = round(
        min(max(score, 0.0), 1.0),
        3
    )

    # ========================================================
    # VERDICT
    # ========================================================

    if score >= 0.60:
        verdict = "SUSPECT"

    elif score >= 0.30:
        verdict = "À_SURVEILLER"

    else:
        verdict = "NORMAL"

    return {
        "score": score,
        "verdict": verdict,
        "anomalies": anomalies,
        "explication": analyse.get(
            "explication",
            ""
        )
    }