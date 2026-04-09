package com.rna.agentservice.config;

import com.rna.agentservice.service.ChatService;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.tool.ToolProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class ChatServiceConfig {

    @Autowired
    private OpenAiChatModel openAiChatModel;
    @Autowired
    private OpenAiStreamingChatModel openAiStreamingChatModel;
    @Autowired
    private ChatMemoryProvider chatMemoryProvider;
    @Autowired
    private ContentRetriever contentRetriever;
    @Autowired
    private ToolProvider toolProvider;

    @Bean
    public ChatService chatService() {
        return AiServices.builder(ChatService.class)
                .chatModel(openAiChatModel)
                .streamingChatModel(openAiStreamingChatModel)
                .chatMemoryProvider(chatMemoryProvider)
                .contentRetriever(contentRetriever)
                .toolProvider(toolProvider)
                .build();
    }
}
