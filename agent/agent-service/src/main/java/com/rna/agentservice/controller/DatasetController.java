package com.rna.agentservice.controller;

import com.rna.agentservice.entity.Dataset;
import com.rna.agentservice.service.DatasetService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/datasets")
public class DatasetController {

    @Autowired
    private DatasetService datasetService;

    // Create dataset
    @PostMapping
    public Dataset create(@RequestBody Dataset dataset) {
        datasetService.save(dataset);
        return dataset;
    }

    // Get all datasets
    @GetMapping
    public List<Dataset> getAll() {
        return datasetService.list();
    }

    // Get dataset by ID
    @GetMapping("/{id}")
    public Dataset getById(@PathVariable Long id) {
        return datasetService.getById(id);
    }

    // Update dataset
    @PutMapping("/{id}")
    public Dataset update(@PathVariable Long id, @RequestBody Dataset dataset) {
        dataset.setId(id);
        datasetService.updateById(dataset);
        return dataset;
    }

    // Delete dataset
    @DeleteMapping("/{id}")
    public void delete(@PathVariable Long id) {
        datasetService.removeById(id);
    }
}
