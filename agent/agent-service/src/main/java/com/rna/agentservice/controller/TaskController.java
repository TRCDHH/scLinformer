package com.rna.agentservice.controller;

import com.rna.agentservice.entity.Task;
import com.rna.agentservice.service.TaskService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/tasks")
public class TaskController {

    @Autowired
    private TaskService taskService;

    // Create task
    @PostMapping
    public Task create(@RequestBody Task task) {
        taskService.save(task);
        return task;
    }

    // Get all tasks
    @GetMapping
    public List<Task> getAll() {
        return taskService.list();
    }

    // Get task by ID
    @GetMapping("/{id}")
    public Task getById(@PathVariable Long id) {
        return taskService.getById(id);
    }

    // Update task
    @PutMapping("/{id}")
    public Task update(@PathVariable Long id, @RequestBody Task task) {
        task.setId(id);
        taskService.updateById(task);
        return task;
    }

    // Delete task
    @DeleteMapping("/{id}")
    public void delete(@PathVariable Long id) {
        taskService.removeById(id);
    }
}
