from __future__ import annotations

import os

from dotenv import load_dotenv
try:
    from neo4j import GraphDatabase
except ModuleNotFoundError:  # permet les tests purs sans connexion Neo4j
    GraphDatabase = None


load_dotenv()


class Neo4jClient:
    """Client Neo4j minimal partagé par les chargeurs et les signaux."""

    def __init__(self) -> None:
        if GraphDatabase is None:
            raise ModuleNotFoundError(
                "Le package neo4j n'est pas installé. Lancez : pip install neo4j"
            )

        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        if not all([self.uri, self.user, self.password]):
            raise ValueError(
                "Les paramètres NEO4J_URI, NEO4J_USER et NEO4J_PASSWORD "
                "sont absents du fichier .env."
            )

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )

    def verifier_connexion(self) -> None:
        self.driver.verify_connectivity()
        print("Connexion à Neo4j réussie.")

    def compter_dossiers(self) -> int:
        records, _, _ = self.driver.execute_query(
            "MATCH (d:Dossier) RETURN count(d) AS nombre_dossiers",
            database_=self.database,
        )
        return int(records[0]["nombre_dossiers"])

    def fermer(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jClient":
        self.verifier_connexion()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.fermer()
