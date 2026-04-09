package com.rna.agentservice.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.util.Date;

@Data
@TableName("task")
public class Task {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String description;
    private Date startTime;
    private Date completeTime;
    private Long datasetId;
    private String modelWeightPath;
    private String testResultPath;
    private String result;
    private Date createTime;
    private Date updateTime;
}