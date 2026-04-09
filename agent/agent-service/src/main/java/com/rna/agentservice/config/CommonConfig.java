package com.rna.agentservice.config;

import dev.langchain4j.community.store.embedding.redis.RedisEmbeddingStore;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.memory.chat.ChatMemoryStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class CommonConfig {

    @Autowired
    private ChatMemoryStore redisChatMemoryStore;

    @Autowired
    private RedisEmbeddingStore redisEmbeddingStore;

    @Bean
    public ChatMemoryProvider chatMemoryProvider() {
        ChatMemoryProvider provider = new ChatMemoryProvider() {
            @Override
            public ChatMemory get(Object memoryId) {
                return MessageWindowChatMemory.builder()
                        .id(memoryId)
                        .maxMessages(20)
                        .chatMemoryStore(redisChatMemoryStore)
                        .build();
            }
        };
        return provider;
    }

    // Vector database operation object
    @Bean
    public EmbeddingStore myEmbeddingStore() {
        // Load documents
        List<Document> documents =ClassPathDocumentLoader.loadDocuments("content");
        // Custom document splitter
        DocumentSplitter ds = DocumentSplitters.recursive(500, 100);
        // Build EmbedingStoreIngestor object, complete text data splitting, vectorization, storage
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .embeddingStore(redisEmbeddingStore)
                .documentSplitter(ds)
//                .embeddingModel(embeddingModel)
                .build();
        // Load RAG data and complete subsequent operations
        ingestor.ingest(documents);
        return redisEmbeddingStore;
    }

    // Vector database retrieval object
    @Bean
    public ContentRetriever contentRetriever() {
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(redisEmbeddingStore)
                .minScore(0.5)
                .maxResults(3)
//                .embeddingModel(embeddingModel)
                .build();
        return retriever;
    }
}
