package com.rna.agentservice.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.util.Date;

@Data
@TableName("dataset")
public class Dataset {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String name;
    private String description;
    private Date uploadTime;
    private String status;
    private String datasetPath;
    private String processedPath;
    private Date createTime;
    private Date updateTime;
}