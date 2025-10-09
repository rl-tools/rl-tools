import { TrajectoryPlayer } from "./TrajectoryPlayer.js";

export class TableExplorer {
    constructor(index, options) {
        this.options = options || {};
        this.container = document.createElement('div');
        this.container.classList.add('table-explorer');
        
        // Store runs data
        this.runs = [];
        this.filteredRuns = [];
        this.sortColumn = 'experiment';
        this.sortDirection = 'desc';
        this.allDataLoaded = false;
        this.initialLoadCount = 100;
        this.commitMessages = {};
        this.selectedRuns = new Set(); // Track selected runs by their experiment ID
        this.onSelectionChange = options.onSelectionChange || null; // Callback when selection changes
        
        // Filters
        this.filters = {
            source: '',
            algorithm: '',
            environment: '',
            experiment: '',
            commit: ''
        };
        
        this.loading_text = document.createElement('div');
        this.loading_text.innerHTML = 'Loading runs...';
        this.container.appendChild(this.loading_text);
        
        // Create initialization promise
        this.initialized = new Promise((resolve) => {
            this.resolveInitialized = resolve;
        });
        
        // Load commit messages and initialize
        this.loadCommitMessages()
            .catch(() => { 
                this.commitMessages = {}; 
            })
            .finally(() => {
                this.initialize();
            });
    }
    
    async loadCommitMessages() {
        try {
            const response = await fetch('./commit_messages.json');
            if (response.ok) {
                this.commitMessages = await response.json();
            } else {
                this.commitMessages = {};
            }
        } catch (e) {
            this.commitMessages = {};
        }
    }
    
    async initialize() {
        this.runs = await this.loadRunsData(this.initialLoadCount);
        this.filteredRuns = [...this.runs];
        this.loading_text.style.display = 'none';
        this.render();
        if (this.resolveInitialized) {
            this.resolveInitialized();
        }
    }
    
    async loadRunsData(limit = null) {
        const runs = [];
        const runList = window.idx.run_list;
        
        // First pass: Create runData objects for all runs
        for (let i = 0; i < runList.length; i++) {
            const run = runList[i];
            const runData = {
                source: run.config._source || 'unknown',
                experiment: run.config.experiment,
                commit: run.config.commit,
                algorithm: run.config.population.algorithm || 'N/A',
                environment: run.config.population.environment || 'N/A',
                seed: run.config.seed,
                hasUi: !!run.ui_jsm,
                hasSteps: Object.keys(run.steps || {}).length > 0,
                stepsCount: Object.keys(run.steps || {}).length,
                run: run,
                returnStats: null,
                detailedDataLoaded: false
            };
            runs.push(runData);
        }
        
        // Sort by experiment (timestamp) descending to get most recent runs first
        runs.sort((a, b) => {
            const valA = a.experiment.toLowerCase();
            const valB = b.experiment.toLowerCase();
            return valB.localeCompare(valA); // Descending order
        });
        
        // Second pass: Load detailed data for the first N most recent runs
        const loadLimit = limit || runs.length;
        for (let i = 0; i < Math.min(loadLimit, runs.length); i++) {
            const runData = runs[i];
            if (runData.run.return) {
                try {
                    const returnData = await (await fetch(runData.run.return)).json();
                    if (returnData && returnData.length > 0) {
                        const final = returnData[returnData.length - 1];
                        runData.returnStats = {
                            final_step: final.step,
                            final_return_mean: final.returns_mean,
                            final_return_std: final.returns_std,
                            episode_length_mean: final.episode_length_mean
                        };
                        runData.detailedDataLoaded = true;
                    }
                } catch (e) {
                    console.warn(`Failed to load return data for run:`, e);
                }
            }
        }
        
        if (!limit || limit >= runs.length) {
            this.allDataLoaded = true;
        }
        
        return runs;
    }
    
    async loadRemainingData() {
        const loadingMessage = document.createElement('div');
        loadingMessage.style.padding = '10px';
        loadingMessage.style.textAlign = 'center';
        loadingMessage.style.backgroundColor = '#ffffcc';
        loadingMessage.innerHTML = `<strong>Loading detailed data for ${this.runs.length - this.initialLoadCount} remaining runs...</strong>`;
        this.container.insertBefore(loadingMessage, this.container.firstChild);
        
        // Load remaining runs
        for (let i = this.initialLoadCount; i < this.runs.length; i++) {
            const runData = this.runs[i];
            if (!runData.detailedDataLoaded && runData.run.return) {
                try {
                    const returnData = await (await fetch(runData.run.return)).json();
                    if (returnData && returnData.length > 0) {
                        const final = returnData[returnData.length - 1];
                        runData.returnStats = {
                            final_step: final.step,
                            final_return_mean: final.returns_mean,
                            final_return_std: final.returns_std,
                            episode_length_mean: final.episode_length_mean
                        };
                        runData.detailedDataLoaded = true;
                    }
                } catch (e) {
                    console.warn(`Failed to load return data for run:`, e);
                }
            }
        }
        
        this.allDataLoaded = true;
        this.container.removeChild(loadingMessage);
        this.render();
    }
    
    render() {
        this.container.innerHTML = '';
        
        // Create controls
        const controls = this.createControls();
        this.container.appendChild(controls);
        
        // Create stats summary
        const summary = this.createSummary();
        this.container.appendChild(summary);
        
        // Create table
        const table = this.createTable();
        this.container.appendChild(table);
    }
    
    createControls() {
        const controls = document.createElement('div');
        controls.classList.add('table-explorer-controls');
        controls.style.marginBottom = '20px';
        controls.style.padding = '15px';
        controls.style.backgroundColor = '#f5f5f5';
        controls.style.borderRadius = '5px';
        
        const title = document.createElement('h3');
        title.textContent = 'Filters';
        title.style.marginTop = '0';
        controls.appendChild(title);
        
        const filtersContainer = document.createElement('div');
        filtersContainer.style.display = 'grid';
        filtersContainer.style.gridTemplateColumns = 'repeat(auto-fit, minmax(200px, 1fr))';
        filtersContainer.style.gap = '10px';
        
        // Get unique values for each filter
        const uniqueValues = {
            source: [...new Set(this.runs.map(r => r.source))].sort(),
            algorithm: [...new Set(this.runs.map(r => r.algorithm))].sort(),
            environment: [...new Set(this.runs.map(r => r.environment))].sort(),
            experiment: [...new Set(this.runs.map(r => r.experiment))].sort().reverse(),
            commit: [...new Set(this.runs.map(r => r.commit))].sort()
        };
        
        // Create filter dropdowns
        for (const [key, values] of Object.entries(uniqueValues)) {
            const filterGroup = document.createElement('div');
            
            const label = document.createElement('label');
            label.textContent = key.charAt(0).toUpperCase() + key.slice(1) + ':';
            label.style.display = 'block';
            label.style.marginBottom = '5px';
            label.style.fontWeight = 'bold';
            
            const select = document.createElement('select');
            select.style.width = '100%';
            select.style.padding = '5px';
            
            const allOption = document.createElement('option');
            allOption.value = '';
            allOption.textContent = `All (${this.runs.filter(r => !this.filters[key] || r[key].includes(this.filters[key])).length})`;
            select.appendChild(allOption);
            
            for (const value of values) {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = value;
                select.appendChild(option);
            }
            
            select.value = this.filters[key];
            select.addEventListener('change', () => {
                this.filters[key] = select.value;
                this.applyFilters();
            });
            
            filterGroup.appendChild(label);
            filterGroup.appendChild(select);
            filtersContainer.appendChild(filterGroup);
        }
        
        controls.appendChild(filtersContainer);
        
        // Reset button
        const resetButton = document.createElement('button');
        resetButton.textContent = 'Reset Filters';
        resetButton.style.marginTop = '10px';
        resetButton.addEventListener('click', () => {
            this.filters = {
                source: '',
                algorithm: '',
                environment: '',
                experiment: '',
                commit: ''
            };
            this.render();
        });
        controls.appendChild(resetButton);
        
        return controls;
    }
    
    createSummary() {
        const summary = document.createElement('div');
        summary.style.marginBottom = '10px';
        summary.style.fontSize = '14px';
        summary.style.color = '#666';
        
        const detailedCount = this.runs.filter(r => r.detailedDataLoaded).length;
        let summaryText = `Showing <strong>${this.filteredRuns.length}</strong> of <strong>${this.runs.length}</strong> runs`;
        
        if (!this.allDataLoaded) {
            summaryText += ` (detailed data loaded for <strong>${detailedCount}</strong>)`;
        }
        
        summary.innerHTML = summaryText;
        
        // Add "Load All" button if not all data is loaded
        if (!this.allDataLoaded) {
            const loadAllButton = document.createElement('button');
            loadAllButton.textContent = `Load Detailed Data for Remaining ${this.runs.length - detailedCount} Runs`;
            loadAllButton.style.marginLeft = '15px';
            loadAllButton.style.padding = '8px 16px';
            loadAllButton.style.backgroundColor = '#4CAF50';
            loadAllButton.style.color = 'white';
            loadAllButton.style.border = 'none';
            loadAllButton.style.borderRadius = '4px';
            loadAllButton.style.cursor = 'pointer';
            loadAllButton.style.fontSize = '14px';
            loadAllButton.addEventListener('click', () => this.loadRemainingData());
            loadAllButton.addEventListener('mouseenter', () => {
                loadAllButton.style.backgroundColor = '#45a049';
            });
            loadAllButton.addEventListener('mouseleave', () => {
                loadAllButton.style.backgroundColor = '#4CAF50';
            });
            summary.appendChild(loadAllButton);
        }
        
        return summary;
    }
    
    createTable() {
        const tableContainer = document.createElement('div');
        tableContainer.style.overflowX = 'auto';
        
        const table = document.createElement('table');
        table.classList.add('table-explorer-table');
        table.style.width = '100%';
        table.style.borderCollapse = 'collapse';
        table.style.fontSize = '14px';
        
        // Create header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        headerRow.style.backgroundColor = '#333';
        headerRow.style.color = 'white';
        
        const columns = [
            { key: 'select', label: '✓', width: '40px' },
            { key: 'source', label: 'Source', width: '80px' },
            { key: 'experiment', label: 'Experiment', width: '150px' },
            { key: 'commit', label: 'Commit', width: '300px' },
            { key: 'algorithm', label: 'Algorithm', width: '100px' },
            { key: 'environment', label: 'Environment', width: '120px' },
            { key: 'seed', label: 'Seed', width: '60px' },
            { key: 'final_return_mean', label: 'Final Return', width: '100px' },
            { key: 'final_return_std', label: 'Return Std', width: '100px' },
            { key: 'final_step', label: 'Steps', width: '100px' },
            { key: 'stepsCount', label: 'Checkpoints', width: '100px' },
            { key: 'actions', label: 'Actions', width: '150px' }
        ];
        
        for (const col of columns) {
            const th = document.createElement('th');
            th.textContent = col.label;
            th.style.padding = '10px';
            th.style.textAlign = 'left';
            th.style.cursor = (col.key !== 'actions' && col.key !== 'select') ? 'pointer' : 'default';
            th.style.borderBottom = '2px solid #ddd';
            if (col.width) th.style.minWidth = col.width;
            
            if (col.key !== 'actions' && col.key !== 'select') {
                th.addEventListener('click', () => this.sortBy(col.key));
                
                if (this.sortColumn === col.key) {
                    const arrow = this.sortDirection === 'asc' ? ' ↑' : ' ↓';
                    th.textContent += arrow;
                }
            }
            
            headerRow.appendChild(th);
        }
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Create body
        const tbody = document.createElement('tbody');
        
        for (const runData of this.filteredRuns) {
            const row = document.createElement('tr');
            row.style.borderBottom = '1px solid #eee';
            row.addEventListener('mouseenter', () => row.style.backgroundColor = '#f9f9f9');
            row.addEventListener('mouseleave', () => row.style.backgroundColor = '');
            
            // Create a unique ID for this run
            const runId = `${runData.experiment}_${runData.seed}`;
            
            // Checkbox
            const tdCheckbox = document.createElement('td');
            tdCheckbox.style.padding = '8px 10px';
            tdCheckbox.style.textAlign = 'center';
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = this.selectedRuns.has(runId);
            checkbox.addEventListener('change', () => {
                if (checkbox.checked) {
                    this.selectedRuns.add(runId);
                } else {
                    this.selectedRuns.delete(runId);
                }
                if (this.onSelectionChange) {
                    this.onSelectionChange(this.getSelectedRuns());
                }
            });
            tdCheckbox.appendChild(checkbox);
            row.appendChild(tdCheckbox);
            
            // Source
            const tdSource = document.createElement('td');
            tdSource.textContent = runData.source;
            tdSource.style.padding = '8px 10px';
            const sourceBadge = document.createElement('span');
            sourceBadge.textContent = runData.source;
            sourceBadge.style.padding = '2px 8px';
            sourceBadge.style.borderRadius = '3px';
            sourceBadge.style.fontSize = '12px';
            sourceBadge.style.backgroundColor = runData.source === 'curated' ? '#4CAF50' : '#2196F3';
            sourceBadge.style.color = 'white';
            tdSource.innerHTML = '';
            tdSource.appendChild(sourceBadge);
            row.appendChild(tdSource);
            
            // Experiment
            const tdExp = document.createElement('td');
            tdExp.textContent = runData.experiment;
            tdExp.style.padding = '8px 10px';
            tdExp.style.fontSize = '12px';
            row.appendChild(tdExp);
            
            // Commit
            const tdCommit = document.createElement('td');
            tdCommit.style.padding = '8px 10px';
            tdCommit.style.fontSize = '12px';
            
            const commitHash = document.createElement('span');
            commitHash.textContent = runData.commit.substring(0, 7);
            commitHash.style.fontFamily = 'monospace';
            commitHash.style.fontWeight = 'bold';
            commitHash.style.marginRight = '8px';
            tdCommit.appendChild(commitHash);
            
            // Add commit message if available
            const shortHash = runData.commit.substring(0, 7);
            if (this.commitMessages[shortHash]) {
                const commitMsg = document.createElement('span');
                commitMsg.textContent = this.commitMessages[shortHash];
                commitMsg.style.color = '#666';
                commitMsg.style.fontSize = '11px';
                tdCommit.appendChild(commitMsg);
                tdCommit.title = this.commitMessages[shortHash]; // Full message on hover
            }
            
            row.appendChild(tdCommit);
            
            // Algorithm
            const tdAlgo = document.createElement('td');
            tdAlgo.textContent = runData.algorithm;
            tdAlgo.style.padding = '8px 10px';
            tdAlgo.style.fontWeight = 'bold';
            row.appendChild(tdAlgo);
            
            // Environment
            const tdEnv = document.createElement('td');
            tdEnv.textContent = runData.environment;
            tdEnv.style.padding = '8px 10px';
            row.appendChild(tdEnv);
            
            // Seed
            const tdSeed = document.createElement('td');
            tdSeed.textContent = runData.seed;
            tdSeed.style.padding = '8px 10px';
            tdSeed.style.textAlign = 'center';
            row.appendChild(tdSeed);
            
            // Final Return Mean
            const tdReturn = document.createElement('td');
            if (runData.returnStats) {
                tdReturn.textContent = runData.returnStats.final_return_mean.toFixed(2);
            } else {
                tdReturn.textContent = 'N/A';
                tdReturn.style.color = '#999';
            }
            tdReturn.style.padding = '8px 10px';
            tdReturn.style.textAlign = 'right';
            row.appendChild(tdReturn);
            
            // Final Return Std
            const tdStd = document.createElement('td');
            if (runData.returnStats) {
                tdStd.textContent = runData.returnStats.final_return_std.toFixed(2);
            } else {
                tdStd.textContent = 'N/A';
                tdStd.style.color = '#999';
            }
            tdStd.style.padding = '8px 10px';
            tdStd.style.textAlign = 'right';
            row.appendChild(tdStd);
            
            // Steps
            const tdSteps = document.createElement('td');
            if (runData.returnStats) {
                tdSteps.textContent = runData.returnStats.final_step.toLocaleString();
            } else {
                tdSteps.textContent = 'N/A';
                tdSteps.style.color = '#999';
            }
            tdSteps.style.padding = '8px 10px';
            tdSteps.style.textAlign = 'right';
            row.appendChild(tdSteps);
            
            // Checkpoints
            const tdCheckpoints = document.createElement('td');
            tdCheckpoints.textContent = runData.stepsCount;
            tdCheckpoints.style.padding = '8px 10px';
            tdCheckpoints.style.textAlign = 'center';
            row.appendChild(tdCheckpoints);
            
            // Actions
            const tdActions = document.createElement('td');
            tdActions.style.padding = '8px 10px';
            
            if (runData.hasUi && runData.hasSteps) {
                const playButton = document.createElement('button');
                playButton.textContent = '▶ Play';
                playButton.style.marginRight = '5px';
                playButton.style.padding = '4px 8px';
                playButton.style.fontSize = '12px';
                playButton.addEventListener('click', () => this.playTrajectory(runData));
                tdActions.appendChild(playButton);
            }
            
            const detailsButton = document.createElement('button');
            detailsButton.textContent = 'Details';
            detailsButton.style.padding = '4px 8px';
            detailsButton.style.fontSize = '12px';
            detailsButton.addEventListener('click', () => this.showDetails(runData));
            tdActions.appendChild(detailsButton);
            
            row.appendChild(tdActions);
            tbody.appendChild(row);
        }
        
        table.appendChild(tbody);
        tableContainer.appendChild(table);
        
        return tableContainer;
    }
    
    applyFilters() {
        this.filteredRuns = this.runs.filter(run => {
            for (const [key, value] of Object.entries(this.filters)) {
                if (value && !run[key].includes(value)) {
                    return false;
                }
            }
            return true;
        });
        
        this.sortData();
        this.render();
    }
    
    sortBy(column) {
        if (this.sortColumn === column) {
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortColumn = column;
            this.sortDirection = 'desc';
        }
        
        this.sortData();
        this.render();
    }
    
    sortData() {
        this.filteredRuns.sort((a, b) => {
            let valA, valB;
            
            if (this.sortColumn === 'final_return_mean' || this.sortColumn === 'final_return_std' || this.sortColumn === 'final_step') {
                valA = a.returnStats ? a.returnStats[this.sortColumn] : -Infinity;
                valB = b.returnStats ? b.returnStats[this.sortColumn] : -Infinity;
            } else {
                valA = a[this.sortColumn];
                valB = b[this.sortColumn];
            }
            
            if (typeof valA === 'string') {
                valA = valA.toLowerCase();
                valB = valB.toLowerCase();
            }
            
            if (valA < valB) return this.sortDirection === 'asc' ? -1 : 1;
            if (valA > valB) return this.sortDirection === 'asc' ? 1 : -1;
            return 0;
        });
    }
    
    playTrajectory(runData) {
        const run = runData.run;
        const steps = Object.keys(run.steps).sort();
        const latestStep = steps[steps.length - 1];
        const step = run.steps[latestStep];
        
        if (!step.trajectories && !step.trajectories_compressed) {
            alert('No trajectories available for this run');
            return;
        }
        
        // Create modal for trajectory player
        const modal = document.createElement('div');
        modal.style.position = 'fixed';
        modal.style.top = '0';
        modal.style.left = '0';
        modal.style.width = '100%';
        modal.style.height = '100%';
        modal.style.backgroundColor = 'rgba(0,0,0,0.8)';
        modal.style.zIndex = '1000';
        modal.style.display = 'flex';
        modal.style.alignItems = 'center';
        modal.style.justifyContent = 'center';
        
        const modalContent = document.createElement('div');
        modalContent.style.backgroundColor = 'white';
        modalContent.style.padding = '20px';
        modalContent.style.borderRadius = '5px';
        modalContent.style.maxWidth = '90%';
        modalContent.style.maxHeight = '90%';
        modalContent.style.overflow = 'auto';
        
        const closeButton = document.createElement('button');
        closeButton.textContent = 'Close';
        closeButton.style.marginBottom = '10px';
        closeButton.addEventListener('click', () => document.body.removeChild(modal));
        modalContent.appendChild(closeButton);
        
        const playerContainer = document.createElement('div');
        const trajectoryPlayer = new TrajectoryPlayer(run.ui_jsm);
        playerContainer.appendChild(trajectoryPlayer.getCanvas());
        modalContent.appendChild(playerContainer);
        
        modal.appendChild(modalContent);
        document.body.appendChild(modal);
        
        const trajectoryPath = step.trajectories_compressed || step.trajectories;
        trajectoryPlayer.playTrajectories(trajectoryPath);
    }
    
    showDetails(runData) {
        const run = runData.run;
        
        // Create modal for details
        const modal = document.createElement('div');
        modal.style.position = 'fixed';
        modal.style.top = '0';
        modal.style.left = '0';
        modal.style.width = '100%';
        modal.style.height = '100%';
        modal.style.backgroundColor = 'rgba(0,0,0,0.8)';
        modal.style.zIndex = '1000';
        modal.style.display = 'flex';
        modal.style.alignItems = 'center';
        modal.style.justifyContent = 'center';
        
        const modalContent = document.createElement('div');
        modalContent.style.backgroundColor = 'white';
        modalContent.style.padding = '20px';
        modalContent.style.borderRadius = '5px';
        modalContent.style.maxWidth = '600px';
        modalContent.style.maxHeight = '80%';
        modalContent.style.overflow = 'auto';
        
        const closeButton = document.createElement('button');
        closeButton.textContent = 'Close';
        closeButton.addEventListener('click', () => document.body.removeChild(modal));
        modalContent.appendChild(closeButton);
        
        const title = document.createElement('h2');
        title.textContent = 'Run Details';
        modalContent.appendChild(title);
        
        const details = document.createElement('pre');
        details.style.backgroundColor = '#f5f5f5';
        details.style.padding = '10px';
        details.style.borderRadius = '3px';
        details.style.overflow = 'auto';
        details.textContent = JSON.stringify({
            source: runData.source,
            experiment: runData.experiment,
            commit: runData.commit,
            algorithm: runData.algorithm,
            environment: runData.environment,
            seed: runData.seed,
            population: run.config.population,
            steps: Object.keys(run.steps || {}),
            returnStats: runData.returnStats
        }, null, 2);
        modalContent.appendChild(details);
        
        modal.appendChild(modalContent);
        document.body.appendChild(modal);
    }
    
    getContainer() {
        return this.container;
    }
    
    getSelectedRuns() {
        // Return the actual run objects for the selected run IDs
        if (!this.runs || !Array.isArray(this.runs)) {
            return [];
        }
        return this.runs.filter(runData => {
            const runId = `${runData.experiment}_${runData.seed}`;
            return this.selectedRuns.has(runId);
        });
    }
    
    setSelectedRuns(runDataArray) {
        // Clear and set new selections
        this.selectedRuns.clear();
        
        if (!runDataArray || !Array.isArray(runDataArray)) {
            return;
        }
        
        for (const runData of runDataArray) {
            const runId = `${runData.experiment}_${runData.seed}`;
            this.selectedRuns.add(runId);
        }
        this.render();
        if (this.onSelectionChange) {
            this.onSelectionChange(this.getSelectedRuns());
        }
    }
}


