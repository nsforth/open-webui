from typing import Dict, List, Optional, Any
from apscheduler.schedulers.background import BackgroundScheduler
from langchain_community.retrievers import BM25Retriever
import logging

from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.models.knowledge import Knowledges, KnowledgeUserModel
from open_webui.retrieval.lexical.stemmers import BilingualStemmer

log = logging.getLogger(__name__)

RETRIVERS_UPDATE_INTERVAL = 1 # Minutes

class Retriever:    
    def __init__(self, collection_id: str, collection_name: str, 
                 collection_updated_at: int, bm25: BM25Retriever):
        self.id: str = collection_id
        self.name: str = collection_name
        self.updated_at: int = collection_updated_at
        self.bm25: Optional[Any] = bm25

class Retrievers:    
    _instance: Optional['Retrievers'] = None
    _initialized: bool = False

    def __new__(cls) -> 'Retrievers':
        if cls._instance is None:
            cls._instance = super(Retrievers, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:        
        if not Retrievers._initialized:            
            self._retrievers: Dict[str, Retriever] = {}  # Stores Retriever objects by their ID
            self._stemmer = BilingualStemmer(additional_language="russian")
            self._scheduler: BackgroundScheduler = BackgroundScheduler()
            self._start_periodic_update()
            Retrievers._initialized = True
            log.info("Lexical Retrievers singleton initialized")

    def _start_periodic_update(self) -> None:        
        self._scheduler.add_job(self._scheduled_update, 'interval', minutes=RETRIVERS_UPDATE_INTERVAL)
        self._scheduler.start()
        log.info(f"Periodic update scheduler started (interval: {RETRIVERS_UPDATE_INTERVAL} minute)")

    def _scheduled_update(self) -> None:        
        log.info(f"Performing lexical retrievers scheduled update")        
        kbs_to_update: List[KnowledgeUserModel] = self.sync_retrievers(Knowledges.get_knowledge_bases())
        if kbs_to_update:            
            self.update_retrievers(kbs_to_update)

    def sync_retrievers(self, incoming: List[KnowledgeUserModel]) -> List[KnowledgeUserModel]:
        kbs_to_update: List[KnowledgeUserModel] = []
        
        incoming_dict: Dict[str, KnowledgeUserModel] = {col.id: col for col in incoming}
        
        for retriever_id, retriever in list(self._retrievers.items()):
            if retriever_id in incoming_dict:
                incoming_kb: KnowledgeUserModel = incoming_dict[retriever_id]                
                if incoming_kb.updated_at > retriever.updated_at:
                    kbs_to_update.append(incoming_kb)
                    log.info(f"Lexical retriever for knowledge base {retriever_id} ({retriever.name}) marked for update: "
                                f"incoming timestamp {incoming_kb.updated_at} > current {retriever.updated_at}")
            else:                
                del self._retrievers[retriever_id]
                log.info(f"Lexical retriever {retriever_id} removed")
        
        for kb in incoming:
            if kb.id not in self._retrievers.keys():
                kbs_to_update.append(kb)
                log.info(f"Lexical retriever {kb.id} created")

        if kbs_to_update:
            log.info(f"Lexical retrievers sync completed: {len(kbs_to_update)} retrievers need updating")
        else:
            log.info("Lexical retrievers sync completed: no retrievers need updating")
        
        return kbs_to_update

    def update_retrievers(self, kbs: List[KnowledgeUserModel]) -> None:
        for kb in kbs:
            collection_id = kb.id
            collection_name = kb.name
            updated_at = kb.updated_at

            collection_result = VECTOR_DB_CLIENT.get(collection_id)

            bm25_retriever = BM25Retriever.from_texts(
                texts=collection_result.documents[0],
                metadatas=collection_result.metadatas[0],
                preprocess_func=self._stemmer
            )            

            self._retrievers[collection_id] = Retriever(
                    collection_id=collection_id,
                    collection_name=collection_name,
                    collection_updated_at=updated_at,
                    bm25=bm25_retriever
                )
               
            if collection_id in self._retrievers:                                
                log.info(f"Updated lexical retriever: {collection_id} ({collection_name})")
            else:                                
                log.info(f"Created new lexical retriever: {collection_id} ({collection_name})")
    
    def get_retriever_by_collection_name(self, collection_name: str, k: int) -> BM25Retriever:
        retriever: Retriever = self._retrievers[collection_name]
        bm25: BM25Retriever = retriever.bm25
        bm25.k = k
        return bm25

LEXICAL_RETRIVERS_INSTANCE = Retrievers()