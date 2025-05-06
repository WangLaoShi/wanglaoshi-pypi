// 调试模式配置
const DEBUG = true;

// 工具函数
const utils = {
    log: function(message, data = null) {
        if (DEBUG) {
            console.log(`[Analyzer] ${message}`, data || '');
        }
    },
    
    error: function(message, error = null) {
        console.error(`[Analyzer Error] ${message}`, error || '');
    },
    
    safeGet: function(obj, path, defaultValue = null) {
        try {
            const result = path.split('.').reduce((acc, part) => acc && acc[part], obj) ?? defaultValue;
            if (DEBUG) {
                console.log(`[Analyzer] 获取路径 ${path} 的值:`, result);
            }
            return result;
        } catch (e) {
            this.error(`Error accessing path ${path}`, e);
            return defaultValue;
        }
    },
    
    escapeHtml: function(unsafe) {
        if (unsafe == null) return '';
        return unsafe
            .toString()
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    },

    showError: function(element, message) {
        element.innerHTML = `<div class="error-message">${this.escapeHtml(message)}</div>`;
    }
};

// 渲染器类
class ReportRenderer {
    constructor(data) {
        if (!data) {
            throw new Error('渲染器初始化失败：未提供数据');
        }
        this.data = data;
        utils.log('渲染器接收到的数据:', this.data);
        
        // 定义部分和对应的渲染方法
        this.sectionMethods = {
            'basic-info': 'renderBasicInfo',
            'basic-stats': 'renderBasicStats',
            'data-quality': 'renderDataQuality',
            'outlier-analysis': 'renderOutlierAnalysis',
            'correlation-analysis': 'renderCorrelationAnalysis',
            'pca-analysis': 'renderPCAAnalysis',
            'visualization': 'renderVisualization',
            'interpretation': 'renderInterpretation'
        };
        
        this.isRendering = false;
        utils.log('渲染器初始化完成', { sections: Object.keys(this.sectionMethods) });
    }

    async render() {
        if (this.isRendering) {
            utils.log('渲染正在进行中，请等待...');
            return;
        }

        this.isRendering = true;
        utils.log('开始渲染报告', this.data);

        try {
            // 渲染数据集名称
            const datasetNameElement = document.getElementById('dataset-name');
            if (datasetNameElement) {
                const datasetName = utils.safeGet(this.data, 'dataset_name', '未命名数据集');
                datasetNameElement.textContent = `数据集：${datasetName}`;
            }

            // 首先渲染基本信息
            await this.renderBasicInfo();
            utils.log('基本信息渲染完成');

            // 然后渲染基础统计
            await this.renderBasicStats();
            utils.log('基础统计渲染完成');

            // 渲染数据质量分析
            await this.renderDataQuality();
            utils.log('数据质量分析渲染完成');

            // 渲染异常值分析
            await this.renderOutlierAnalysis();
            utils.log('异常值分析渲染完成');

            // 渲染相关性分析
            await this.renderCorrelationAnalysis();
            utils.log('相关性分析渲染完成');

            // 渲染主成分分析
            await this.renderPCAAnalysis();
            utils.log('主成分分析渲染完成');

            // 渲染可视化
            await this.renderVisualization();
            utils.log('可视化渲染完成');

            // 渲染解读
            await this.renderInterpretation();
            utils.log('解读渲染完成');

            utils.log('报告渲染完成');
        } catch (error) {
            utils.error('渲染过程中出错', error);
            throw error;
        } finally {
            this.isRendering = false;
        }
    }

    async renderBasicInfo() {
        const element = document.getElementById('basic-info');
        if (!element) {
            utils.error('未找到基本信息元素');
            return;
        }

        try {
            console.log('开始渲染基本信息');
            console.log('DOM元素:', element);
            console.log('完整数据:', this.data);
            
            const basicInfo = utils.safeGet(this.data, 'interpretations.basic_info', {});
            console.log('获取到的基本信息:', basicInfo);
            console.log('变量详情:', basicInfo.变量详情);
            
            if (!basicInfo || Object.keys(basicInfo).length === 0) {
                console.warn('未找到基本信息数据');
                utils.showError(element, '未找到基本信息数据');
                return;
            }

            let html = `
                <div class="interpretation">
                    <p>${utils.escapeHtml(basicInfo.数据集规模 || '无数据集规模信息')}</p>
                    <p>${utils.escapeHtml(basicInfo.变量类型 || '无变量类型信息')}</p>
                    <p>${utils.escapeHtml(basicInfo.说明 || '无补充说明')}</p>`;
            
            // 检查变量详情是否存在
            if (basicInfo.变量详情) {
                console.log('开始渲染变量详情');
                html += `<h3>变量详情</h3>`;

                // 添加数值型变量
                if (basicInfo.变量详情.数值型变量 && basicInfo.变量详情.数值型变量.length > 0) {
                    console.log('渲染数值型变量:', basicInfo.变量详情.数值型变量);
                    html += `
                        <h4>数值型变量</h4>
                        <ul>`;
                    basicInfo.变量详情.数值型变量.forEach(variable => {
                        html += `
                            <li>
                                <strong>${utils.escapeHtml(variable.name)}</strong>
                                <span class="sample">示例值: ${utils.escapeHtml(variable.sample)}</span>
                            </li>`;
                    });
                    html += `</ul>`;
                }

                // 添加分类型变量
                if (basicInfo.变量详情.分类型变量 && basicInfo.变量详情.分类型变量.length > 0) {
                    console.log('渲染分类型变量:', basicInfo.变量详情.分类型变量);
                    html += `
                        <h4>分类型变量</h4>
                        <ul>`;
                    basicInfo.变量详情.分类型变量.forEach(variable => {
                        html += `
                            <li>
                                <strong>${utils.escapeHtml(variable.name)}</strong>
                                <span class="sample">示例值: ${utils.escapeHtml(variable.sample)}</span>
                            </li>`;
                    });
                    html += `</ul>`;
                }

                // 添加时间型变量
                if (basicInfo.变量详情.时间型变量 && basicInfo.变量详情.时间型变量.length > 0) {
                    console.log('渲染时间型变量:', basicInfo.变量详情.时间型变量);
                    html += `
                        <h4>时间型变量</h4>
                        <ul>`;
                    basicInfo.变量详情.时间型变量.forEach(variable => {
                        html += `
                            <li>
                                <strong>${utils.escapeHtml(variable.name)}</strong>
                                <span class="sample">示例值: ${utils.escapeHtml(variable.sample)}</span>
                            </li>`;
                    });
                    html += `</ul>`;
                }
            } else {
                console.warn('未找到变量详情数据');
            }

            html += `</div>`;
            console.log('生成的HTML:', html);
            
            element.innerHTML = html;
            element.classList.add('rendered');
            console.log('基本信息渲染完成');
        } catch (e) {
            console.error('渲染基本信息时出错:', e);
            this.handleError(element, e);
        }
    }

    async renderBasicStats() {
        const element = document.getElementById('basic-stats');
        if (!element) {
            utils.error('未找到基础统计元素');
            return;
        }

        try {
            console.log('开始渲染基础统计');
            console.log('DOM元素:', element);
            
            const stats = utils.safeGet(this.data, 'basic_stats', []);
            console.log('获取到的统计数据:', stats);
            
            if (!Array.isArray(stats) || stats.length === 0) {
                console.warn('无统计数据可用');
                utils.showError(element, '无统计数据可用');
                return;
            }

            let html = '<div class="interpretation">';

            // 数值型变量统计
            const numericStats = stats.filter(row => {
                const isNumeric = !isNaN(row.mean) && row.mean !== '空值';
                return isNumeric;
            });

            if (numericStats.length > 0) {
                html += `
                    <h3>数值型变量统计</h3>
                    <div class="table-container">
                        <table>
                            <tr>
                                <th>列名</th>
                                <th>计数</th>
                                <th>均值</th>
                                <th>标准差</th>
                                <th>最小值</th>
                                <th>25%分位数</th>
                                <th>中位数</th>
                                <th>75%分位数</th>
                                <th>最大值</th>
                            </tr>`;
                
                numericStats.forEach((row, index) => {
                    console.log(`处理第 ${index + 1} 行数值型数据:`, row);
                    html += `<tr>
                        <td>${utils.escapeHtml(row.列名)}</td>
                        <td>${utils.escapeHtml(row.count)}</td>
                        <td>${utils.escapeHtml(row.mean)}</td>
                        <td>${utils.escapeHtml(row.std)}</td>
                        <td>${utils.escapeHtml(row.min)}</td>
                        <td>${utils.escapeHtml(row['25%'])}</td>
                        <td>${utils.escapeHtml(row['50%'])}</td>
                        <td>${utils.escapeHtml(row['75%'])}</td>
                        <td>${utils.escapeHtml(row.max)}</td>
                    </tr>`;
                });
                html += '</table></div>';
            }

            // 分类型变量统计
            const categoricalStats = stats.filter(row => {
                const isCategorical = isNaN(row.mean) || row.mean === '空值';
                return isCategorical;
            });

            if (categoricalStats.length > 0) {
                html += `
                    <h3>分类型变量统计</h3>
                    <div class="table-container">
                        <table>
                            <tr>
                                <th>列名</th>
                                <th>计数</th>
                                <th>唯一值数量</th>
                                <th>缺失值数量</th>
                                <th>缺失率 (%)</th>
                            </tr>`;
                
                categoricalStats.forEach((row, index) => {
                    console.log(`处理第 ${index + 1} 行分类型数据:`, row);
                    html += `<tr>
                        <td>${utils.escapeHtml(row.列名)}</td>
                        <td>${utils.escapeHtml(row.count)}</td>
                        <td>${utils.escapeHtml(row['唯一值数量'])}</td>
                        <td>${utils.escapeHtml(row['缺失值数量'])}</td>
                        <td>${utils.escapeHtml(row['缺失率 (%)'])}</td>
                    </tr>`;
                });
                html += '</table></div>';
            }

            html += '</div>';
            console.log('生成的HTML:', html);
            
            element.innerHTML = html;
            element.classList.add('rendered');
            console.log('基础统计渲染完成');
        } catch (e) {
            console.error('渲染基础统计时出错:', e);
            this.handleError(element, e);
        }
    }

    async renderDataQuality() {
        const element = document.getElementById('data-quality');
        try {
            const quality = utils.safeGet(this.data, 'interpretations.data_quality', {});
            let html = '';
            
            // 缺失值分析
            const missingAnalysis = utils.safeGet(quality, '缺失值分析', {});
            html += `<h3>缺失值分析</h3>
                <p>${utils.escapeHtml(missingAnalysis.总体情况 || '无缺失值分析数据')}</p>`;
            
            if (Array.isArray(missingAnalysis.details) && missingAnalysis.details.length > 0) {
                html += '<ul>';
                missingAnalysis.details.forEach(detail => {
                    html += `<li>${utils.escapeHtml(detail.变量)}: ${utils.escapeHtml(detail.缺失情况)} - ${utils.escapeHtml(detail.建议)}</li>`;
                });
                html += '</ul>';
            }

            // 异常值分析
            const outlierAnalysis = utils.safeGet(quality, '异常值分析', {});
            html += `<h3>异常值分析</h3>
                <p>${utils.escapeHtml(outlierAnalysis.总体情况 || '无异常值分析数据')}</p>`;
            
            if (Array.isArray(outlierAnalysis.details) && outlierAnalysis.details.length > 0) {
                html += '<ul>';
                outlierAnalysis.details.forEach(detail => {
                    html += `<li>${utils.escapeHtml(detail.变量)}: ${utils.escapeHtml(detail.异常值比例)} - ${utils.escapeHtml(detail.建议)}</li>`;
                });
                html += '</ul>';
            }

            element.innerHTML = html || '<div class="error-message">无数据质量分析数据</div>';
        } catch (e) {
            this.handleError(element, e);
        }
    }

    async renderOutlierAnalysis() {
        const element = document.getElementById('outlier-analysis');
        try {
            const outliers = utils.safeGet(this.data, 'outlier_analysis', []);
            if (!Array.isArray(outliers) || outliers.length === 0) {
                element.innerHTML = '<div class="error-message">无异常值分析数据</div>';
                return;
            }

            let html = '<table><tr><th>列名</th><th>Z-score异常值数量</th><th>IQR异常值数量</th><th>异常值比例 (%)</th></tr>';
            outliers.forEach(row => {
                html += `<tr>
                    <td>${utils.escapeHtml(row.列名)}</td>
                    <td>${utils.escapeHtml(row['Z-score异常值数量'])}</td>
                    <td>${utils.escapeHtml(row['IQR异常值数量'])}</td>
                    <td>${utils.escapeHtml(row['异常值比例 (%)'])}</td>
                </tr>`;
            });
            html += '</table>';
            element.innerHTML = html;
        } catch (e) {
            this.handleError(element, e);
        }
    }

    async renderCorrelationAnalysis() {
        const element = document.getElementById('correlation-analysis');
        try {
            const correlationPlot = utils.safeGet(this.data, 'plots.correlation');
            if (!correlationPlot) {
                element.innerHTML = '<div class="error-message">无相关性分析数据</div>';
                return;
            }
            element.innerHTML = `<div class="plot"><img src="data:image/png;base64,${correlationPlot}" alt="相关性热图"></div>`;
        } catch (e) {
            this.handleError(element, e);
        }
    }

    async renderPCAAnalysis() {
        const element = document.getElementById('pca-analysis');
        try {
            const pca = utils.safeGet(this.data, 'pca_analysis', {});
            if (!pca || !pca.主成分数量) {
                element.innerHTML = '<div class="error-message">无主成分分析数据</div>';
                return;
            }

            const varianceRatio = utils.safeGet(pca, `累计方差贡献率.${pca.建议保留主成分数量-1}`, 0) * 100;
            let html = `
                <p>主成分数量: ${utils.escapeHtml(pca.主成分数量)}</p>
                <p>建议保留主成分数量: ${utils.escapeHtml(pca.建议保留主成分数量)}</p>
                <p>累计方差贡献率: ${varianceRatio.toFixed(2)}%</p>
            `;
            element.innerHTML = html;
        } catch (e) {
            this.handleError(element, e);
        }
    }

    async renderVisualization() {
        const element = document.getElementById('visualization');
        try {
            const plots = utils.safeGet(this.data, 'plots.distribution', {});
            if (!plots || Object.keys(plots).length === 0) {
                element.innerHTML = '<div class="error-message">无可视化数据</div>';
                return;
            }

            let html = '<div class="plots-container">';
            for (const [column, plot] of Object.entries(plots)) {
                html += `
                    <div class="plot">
                        <h3>${utils.escapeHtml(column)} 分布图</h3>
                        <img src="data:image/png;base64,${plot}" alt="${utils.escapeHtml(column)} 分布图">
                    </div>
                `;
            }
            html += '</div>';
            element.innerHTML = html;
        } catch (e) {
            this.handleError(element, e);
        }
    }

    async renderInterpretation() {
        const element = document.getElementById('interpretation');
        try {
            const interpretation = utils.safeGet(this.data, 'interpretations', {});
            if (!interpretation) {
                element.innerHTML = '<div class="error-message">无数据解读</div>';
                return;
            }

            let html = '<div class="interpretation">';
            
            // 处理数据质量分析
            const dataQuality = interpretation.data_quality || {};
            html += '<h3>数据质量分析</h3>';
            
            // 缺失值分析
            if (dataQuality.缺失值分析) {
                html += `<h4>缺失值分析</h4>`;
                html += `<p>${utils.escapeHtml(dataQuality.缺失值分析.总体情况 || '')}</p>`;
                if (dataQuality.缺失值分析.details && dataQuality.缺失值分析.details.length > 0) {
                    html += '<ul>';
                    dataQuality.缺失值分析.details.forEach(detail => {
                        html += `<li>${utils.escapeHtml(detail.变量)}: ${utils.escapeHtml(detail.缺失情况)} - ${utils.escapeHtml(detail.建议)}</li>`;
                    });
                    html += '</ul>';
                }
            }
            
            // 异常值分析
            if (dataQuality.异常值分析) {
                html += `<h4>异常值分析</h4>`;
                html += `<p>${utils.escapeHtml(dataQuality.异常值分析.总体情况 || '')}</p>`;
                if (dataQuality.异常值分析.details && dataQuality.异常值分析.details.length > 0) {
                    html += '<ul>';
                    dataQuality.异常值分析.details.forEach(detail => {
                        html += `<li>${utils.escapeHtml(detail.变量)}: ${utils.escapeHtml(detail.异常值比例)} - ${utils.escapeHtml(detail.建议)}</li>`;
                    });
                    html += '</ul>';
                }
            }
            
            // 重复值分析
            if (dataQuality.重复值分析) {
                html += `<h4>重复值分析</h4>`;
                html += `<p>${utils.escapeHtml(dataQuality.重复值分析.总体情况 || '')}</p>`;
                html += `<p>${utils.escapeHtml(dataQuality.重复值分析.建议 || '')}</p>`;
            }
            
            // 处理统计分析
            const statisticalAnalysis = interpretation.statistical_analysis || {};
            html += '<h3>统计分析</h3>';
            
            // 正态性检验
            if (statisticalAnalysis.正态性检验) {
                html += `<h4>正态性检验</h4>`;
                html += `<p>${utils.escapeHtml(statisticalAnalysis.正态性检验.总体情况 || '')}</p>`;
                if (statisticalAnalysis.正态性检验.details && statisticalAnalysis.正态性检验.details.length > 0) {
                    html += '<ul>';
                    statisticalAnalysis.正态性检验.details.forEach(detail => {
                        html += `<li>${utils.escapeHtml(detail.变量)}: ${utils.escapeHtml(detail.是否正态分布)} - ${utils.escapeHtml(detail.解释)}</li>`;
                    });
                    html += '</ul>';
                }
            }
            
            // 峰度偏度分析
            if (statisticalAnalysis.峰度偏度分析) {
                html += `<h4>峰度偏度分析</h4>`;
                html += `<p>${utils.escapeHtml(statisticalAnalysis.峰度偏度分析.总体情况 || '')}</p>`;
                if (statisticalAnalysis.峰度偏度分析.details && statisticalAnalysis.峰度偏度分析.details.length > 0) {
                    html += '<ul>';
                    statisticalAnalysis.峰度偏度分析.details.forEach(detail => {
                        html += `<li>${utils.escapeHtml(detail.变量)}: 偏度 ${utils.escapeHtml(detail.偏度)} (${utils.escapeHtml(detail.偏度解释)}), 峰度 ${utils.escapeHtml(detail.峰度)} (${utils.escapeHtml(detail.峰度解释)})</li>`;
                    });
                    html += '</ul>';
                }
            }
            
            // 相关性分析
            if (statisticalAnalysis.相关性分析) {
                html += `<h4>相关性分析</h4>`;
                html += `<p>${utils.escapeHtml(statisticalAnalysis.相关性分析.总体情况 || '')}</p>`;
                
                if (statisticalAnalysis.相关性分析.强相关变量 && statisticalAnalysis.相关性分析.强相关变量.length > 0) {
                    html += '<h5>强相关变量</h5><ul>';
                    statisticalAnalysis.相关性分析.强相关变量.forEach(item => {
                        html += `<li>${utils.escapeHtml(item.变量对)}: ${utils.escapeHtml(item.相关系数)} - ${utils.escapeHtml(item.解释)}</li>`;
                    });
                    html += '</ul>';
                }
                
                if (statisticalAnalysis.相关性分析.中等相关变量 && statisticalAnalysis.相关性分析.中等相关变量.length > 0) {
                    html += '<h5>中等相关变量</h5><ul>';
                    statisticalAnalysis.相关性分析.中等相关变量.forEach(item => {
                        html += `<li>${utils.escapeHtml(item.变量对)}: ${utils.escapeHtml(item.相关系数)} - ${utils.escapeHtml(item.解释)}</li>`;
                    });
                    html += '</ul>';
                }
            }
            
            // 主成分分析
            if (statisticalAnalysis.主成分分析) {
                html += `<h4>主成分分析</h4>`;
                html += `<p>${utils.escapeHtml(statisticalAnalysis.主成分分析.总体情况 || '')}</p>`;
                html += `<p>${utils.escapeHtml(statisticalAnalysis.主成分分析.建议保留主成分数 || '')}</p>`;
                html += `<p>${utils.escapeHtml(statisticalAnalysis.主成分分析.解释 || '')}</p>`;
            }
            
            // 处理建议
            const recommendations = interpretation.recommendations || [];
            if (recommendations.length > 0) {
                html += '<h3>建议</h3><ul>';
                recommendations.forEach(rec => {
                    html += `<li><strong>${utils.escapeHtml(rec.类型)}:</strong> ${utils.escapeHtml(rec.问题)} - ${utils.escapeHtml(rec.建议)}</li>`;
                });
                html += '</ul>';
            }
            
            html += '</div>';
            element.innerHTML = html;
            element.classList.add('rendered');
        } catch (e) {
            console.error('渲染解读时出错:', e);
            this.handleError(element, e);
        }
    }

    handleError(element, error) {
        utils.error('渲染错误', error);
        utils.showError(element, `渲染出错: ${error.message}`);
    }
}

// 初始化
document.addEventListener('DOMContentLoaded', async () => {
    try {
        utils.log('初始化报告渲染器');
        
        if (!window.ANALYZER_DATA) {
            throw new Error('未找到分析数据');
        }
        
        // 检查数据结构
        const data = window.ANALYZER_DATA;
        const requiredFields = ['basic_stats', 'interpretations', 'plots'];
        const missingFields = requiredFields.filter(field => !data[field]);
        
        if (missingFields.length > 0) {
            throw new Error(`数据缺少必要字段: ${missingFields.join(', ')}`);
        }
        
        utils.log('数据加载成功，数据结构:', {
            hasBasicStats: Array.isArray(data.basic_stats),
            hasInterpretations: typeof data.interpretations === 'object',
            hasPlots: typeof data.plots === 'object',
            basicStatsLength: data.basic_stats.length,
            interpretationsKeys: Object.keys(data.interpretations),
            plotsKeys: Object.keys(data.plots)
        });
        
        const renderer = new ReportRenderer(data);
        await renderer.render();
    } catch (error) {
        utils.error('初始化失败', error);
        // 显示错误信息在页面上
        const container = document.querySelector('.container');
        if (container) {
            container.innerHTML = `
                <div class="error-message">
                    <h2>报告加载失败</h2>
                    <p>错误信息: ${utils.escapeHtml(error.message)}</p>
                    <p>请检查数据格式是否正确，或联系管理员。</p>
                </div>
            `;
        }
    }
}); 