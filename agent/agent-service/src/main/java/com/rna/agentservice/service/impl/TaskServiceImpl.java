package com.rna.agentservice.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.rna.agentservice.entity.Task;
import com.rna.agentservice.mapper.TaskMapper;
import com.rna.agentservice.service.TaskService;
import org.springframework.stereotype.Service;

@Service
public class TaskServiceImpl extends ServiceImpl<TaskMapper, Task> implements TaskService {
}