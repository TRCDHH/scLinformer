package com.rna.agentservice.controller;

import com.rna.agentservice.service.ChatService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api/agent")
public class ChatController {

    @Autowired
    private ChatService chatService;

    @GetMapping(value = "/chat", produces = "text/html;charset=UTF-8")
    public Flux<String> chat(@RequestParam("memoryId") String memoryId, @RequestParam("message") String message) {
        Flux<String> result = chatService.chat(memoryId, message);
        return result;
    }
}
