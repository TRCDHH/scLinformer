package com.rna.agentservice.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.rna.agentservice.entity.Dataset;
import com.rna.agentservice.mapper.DatasetMapper;
import com.rna.agentservice.service.DatasetService;
import org.springframework.stereotype.Service;

@Service
public class DatasetServiceImpl extends ServiceImpl<DatasetMapper, Dataset> implements DatasetService {
}