import React, { useState, useEffect, useContext } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'react-toastify';
import { 
  Rocket, RefreshCw, X, ChevronDown, Gauge, Crosshair, 
  TrendingUp, Activity, Layers, Zap, AlertCircle, CheckCircle,
  Info, ChevronRight, BarChart2, Percent, Target, Circle
} from 'lucide-react';
import { useAppContext } from '../context/AppContext';

const PredictionForm = ({ onClose }) => {
  const { 
    selectedDataset, 
    selectedTable, 
    selectedModel,
    projectId = "bigdata-sprint",
    bucket = "my-smart-ingest-bucket"
  } = useContext(AppContext);
  
  const { register, handleSubmit, reset, watch, setValue } = useForm();
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionProgress, setPredictionProgress] = useState(0);
  const [predictionStatus, setPredictionStatus] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [formMetadata, setFormMetadata] = useState(null);
  const [showProbs, setShowProbs] = useState(false);
  
  // Simulate fetching form metadata
  useEffect(() => {
    if (!selectedModel) return;
    
    const fetchFormMetadata = async () => {
      try {
        // Simulate API call
        setTimeout(() => {
          const modelPath = selectedModel.fullPath.replace(`gs://${bucket}/`, "");
          
          // Mock metadata response
          const mockMetadata = {
            status: 'success',
            problem_type: 'binary_classification',
            target_column: 'target',
            model_info: {
              training_date: new Date().toISOString(),
              metrics: {
                accuracy: 0.92,
                precision: 0.89,
                recall: 0.91,
                f1: 0.90
              }
            },
            form_fields: {
              age: {
                type: 'number',
                input_type: 'number',
                min: 18,
                max: 100,
                default_value: 30,
                description: 'Age of the individual'
              },
              income: {
                type: 'number',
                input_type: 'number',
                min: 0,
                max: 200000,
                default_value: 50000,
                description: 'Annual income in USD'
              },
              education: {
                type: 'string',
                input_type: 'select',
                options: ['High School', 'Bachelor', 'Master', 'PhD'],
                default_value: 'Bachelor',
                description: 'Highest education level'
              }
            }
          };
          
          setFormMetadata(mockMetadata);
          setShowProbs(mockMetadata.problem_type !== 'regression');
          
          // Set default values for form fields
          Object.entries(mockMetadata.form_fields).forEach(([field, config]) => {
            if (config.default_value !== undefined) {
              setValue(field, config.default_value);
            }
          });
        }, 1000);
      } catch (error) {
        toast.error(
          <div className="flex items-center">
            <AlertCircle className="w-5 h-5 mr-2 text-red-500" />
            Failed to load form metadata: {error.message}
          </div>,
          { className: 'bg-gray-800 text-white' }
        );
      }
    };
    
    fetchFormMetadata();
  }, [selectedModel, bucket, setValue]);
  
  const onSubmit = async (data) => {
    setIsPredicting(true);
    setPredictionProgress(0);
    setPredictionStatus('Preparing prediction request...');
    
    try {
      // Simulate prediction progress
      const simulateProgress = () => {
        return new Promise((resolve) => {
          const intervals = [25, 50, 75, 100];
          intervals.forEach((progress, i) => {
            setTimeout(() => {
              setPredictionProgress(progress);
              setPredictionStatus([
                'Preparing prediction request...',
                'Sending request to prediction service...',
                'Processing prediction...',
                'Prediction completed!'
              ][i]);
              if (progress === 100) resolve();
            }, i * 1000);
          });
        });
      };
      
      await simulateProgress();
      
      // Simulate API response
      setTimeout(() => {
        const modelPath = selectedModel.fullPath.replace(`gs://${bucket}/`, "");
        const problemType = formMetadata.problem_type;
        
        let mockResult;
        if (problemType === 'regression') {
          mockResult = {
            status: 'success',
            prediction: 42.5,
            probabilities: null
          };
        } else if (problemType === 'binary_classification') {
          mockResult = {
            status: 'success',
            prediction: [1],
            probabilities: 0.87
          };
        } else {
          // Multi-class
          mockResult = {
            status: 'success',
            prediction: [2],
            probabilities: [0.1, 0.2, 0.7]
          };
        }
        
        setPredictionResult({
          ...mockResult,
          input_data: data
        });
        
        toast.success(
          <div className="flex items-center">
            <CheckCircle className="w-5 h-5 mr-2 text-green-500" />
            Prediction completed successfully!
          </div>,
          { className: 'bg-gray-800 text-white' }
        );
      }, 1000);
    } catch (error) {
      toast.error(
        <div className="flex items-center">
          <AlertCircle className="w-5 h-5 mr-2 text-red-500" />
          Prediction failed: {error.message}
        </div>,
        { className: 'bg-gray-800 text-white' }
      );
    } finally {
      setIsPredicting(false);
    }
  };
  
  const handleReset = () => {
    reset();
    if (formMetadata) {
      Object.entries(formMetadata.form_fields).forEach(([field, config]) => {
        if (config.default_value !== undefined) {
          setValue(field, config.default_value);
        }
      });
    }
  };
  
  if (!selectedModel || !selectedDataset || !selectedTable) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
        <div className="w-full max-w-4xl p-6 rounded-lg shadow-xl bg-gray-900 text-white">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold">Make Prediction</h2>
            <button 
              onClick={onClose}
              className="p-1 rounded-full hover:bg-gray-700"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="p-4 rounded-md bg-red-900 bg-opacity-30 border border-red-800">
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 mr-2 text-red-400" />
              <span>Missing required information. Please select a model first.</span>
            </div>
          </div>
          <div className="flex justify-center mt-6">
            <button
              onClick={onClose}
              className="px-6 py-2 font-medium rounded-md bg-gray-700 hover:bg-gray-600 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }
  
  if (!formMetadata) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
        <div className="w-full max-w-4xl p-6 rounded-lg shadow-xl bg-gray-900 text-white">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold">Make Prediction</h2>
            <button 
              onClick={onClose}
              className="p-1 rounded-full hover:bg-gray-700"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="flex flex-col items-center justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
            <p>Loading form fields...</p>
          </div>
        </div>
      </div>
    );
  }
  
  if (formMetadata.status !== 'success') {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
        <div className="w-full max-w-4xl p-6 rounded-lg shadow-xl bg-gray-900 text-white">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold">Make Prediction</h2>
            <button 
              onClick={onClose}
              className="p-1 rounded-full hover:bg-gray-700"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="p-4 rounded-md bg-red-900 bg-opacity-30 border border-red-800">
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 mr-2 text-red-400" />
              <span>Failed to load model metadata</span>
            </div>
          </div>
          <div className="flex justify-center mt-6">
            <button
              onClick={onClose}
              className="px-6 py-2 font-medium rounded-md bg-gray-700 hover:bg-gray-600 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="w-full max-w-4xl p-6 rounded-lg shadow-xl bg-gray-900 text-white">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Make Prediction</h2>
          <button 
            onClick={onClose}
            className="p-1 rounded-full hover:bg-gray-700"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        {/* Info Box */}
        <div className="p-4 mb-6 rounded-md bg-gray-800">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="font-semibold">Model:</span> 
              <span className="ml-2 font-mono text-blue-300">{selectedModel.filename}</span>
            </div>
            <div>
              <span className="font-semibold">Problem Type:</span> 
              <span className="ml-2 font-mono text-blue-300">
                {formMetadata.problem_type.replace('_', ' ').title()}
              </span>
            </div>
            <div>
              <span className="font-semibold">Target Column:</span> 
              <span className="ml-2 font-mono text-blue-300">{formMetadata.target_column}</span>
            </div>
            <div>
              <span className="font-semibold">Dataset:</span> 
              <span className="ml-2 font-mono text-blue-300">{selectedDataset}</span>
            </div>
          </div>
        </div>
        
        {/* Model Info Expandable */}
        <div className="mb-6">
          <div className="p-3 rounded-md bg-gray-800">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center justify-between w-full"
            >
              <span className="flex items-center">
                <Info className="w-5 h-5 mr-2 text-blue-400" />
                Model Information
              </span>
              <ChevronDown className={`w-5 h-5 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
            </button>
            
            {showAdvanced && (
              <div className="mt-4 pt-4 border-t border-gray-700">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-gray-400">Training Date</div>
                    <div className="font-medium">
                      {new Date(formMetadata.model_info.training_date).toLocaleDateString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400">Accuracy</div>
                    <div className="font-medium">
                      {formMetadata.model_info.metrics.accuracy.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400">Precision</div>
                    <div className="font-medium">
                      {formMetadata.model_info.metrics.precision.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400">Recall</div>
                    <div className="font-medium">
                      {formMetadata.model_info.metrics.recall.toFixed(3)}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {isPredicting ? (
          <div className="space-y-4">
            <h3 className="flex items-center text-lg font-semibold">
              <Activity className="w-5 h-5 mr-2 text-yellow-400" />
              Prediction in Progress...
            </h3>
            <p className="flex items-center text-sm text-gray-300">
              <Zap className="w-4 h-4 mr-2 text-yellow-400" />
              {predictionStatus}
            </p>
            <div className="w-full h-2 bg-gray-700 rounded-full">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 to-green-500 rounded-full transition-all duration-500"
                style={{ width: `${predictionProgress}%` }}
              ></div>
            </div>
            <div className="text-right text-sm text-gray-400">
              {predictionProgress}% complete
            </div>
          </div>
        ) : predictionResult ? (
          <div className="space-y-6">
            {/* Success Message */}
            <div className="p-4 rounded-md bg-green-900 bg-opacity-30 border border-green-800">
              <div className="flex items-center">
                <CheckCircle className="w-5 h-5 mr-2 text-green-400" />
                <span className="font-semibold">Prediction completed successfully!</span>
              </div>
            </div>
            
            {/* Prediction Result */}
            <div>
              <h3 className="flex items-center mb-4 text-lg font-semibold">
                <Target className="w-5 h-5 mr-2 text-purple-400" />
                Prediction Result
              </h3>
              
              {formMetadata.problem_type === 'regression' ? (
                <div className="p-4 rounded-md bg-gray-800">
                  <div className="text-2xl font-bold text-center mb-2">
                    {typeof predictionResult.prediction === 'number' 
                      ? predictionResult.prediction.toFixed(2)
                      : parseFloat(predictionResult.prediction).toFixed(2)}
                  </div>
                  <div className="text-center text-gray-300">
                    Predicted {formMetadata.target_column.replace('_', ' ').title()}
                  </div>
                  <div className="mt-4 p-3 rounded-md bg-gray-700">
                    <p className="text-sm">
                      The model predicts <span className="font-medium">{predictionResult.prediction.toFixed(2)}</span> for {formMetadata.target_column.replace('_', ' ').title()}
                    </p>
                  </div>
                </div>
              ) : formMetadata.problem_type === 'binary_classification' ? (
                <div className="p-4 rounded-md bg-gray-800">
                  <div className={`text-2xl font-bold text-center mb-2 ${predictionResult.prediction[0] === 1 ? 'text-green-400' : 'text-red-400'}`}>
                    {predictionResult.prediction[0] === 1 ? 'POSITIVE/YES' : 'NEGATIVE/NO'}
                  </div>
                  <div className="text-center text-gray-300">
                    {formMetadata.target_column.replace('_', ' ').title()}
                  </div>
                  <div className="mt-2 flex justify-center">
                    <div className="w-3/4 h-4 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                        style={{ 
                          width: `${predictionResult.prediction[0] === 1 
                            ? (predictionResult.probabilities * 100) 
                            : ((1 - predictionResult.probabilities) * 100)}%` 
                        }}
                      ></div>
                    </div>
                  </div>
                  <div className="mt-2 text-center text-sm text-gray-400">
                    Confidence: {predictionResult.prediction[0] === 1 
                      ? (predictionResult.probabilities * 100).toFixed(1) 
                      : ((1 - predictionResult.probabilities) * 100).toFixed(1)}%
                  </div>
                  <div className="mt-4 p-3 rounded-md bg-gray-700">
                    <p className="text-sm">
                      The model predicts a {predictionResult.prediction[0] === 1 ? 'positive' : 'negative'} outcome for {formMetadata.target_column.replace('_', ' ').title()}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="p-4 rounded-md bg-gray-800">
                  <div className="text-2xl font-bold text-center mb-2 text-blue-400">
                    CLASS {predictionResult.prediction[0]}
                  </div>
                  <div className="text-center text-gray-300">
                    {formMetadata.target_column.replace('_', ' ').title()}
                  </div>
                  <div className="mt-4">
                    {predictionResult.probabilities.map((prob, idx) => (
                      <div key={idx} className="mb-2">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm">Class {idx}</span>
                          <span className="text-sm font-mono">{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-blue-500"
                            style={{ width: `${prob * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            
            {/* Detailed Results */}
            <div className="p-3 rounded-md bg-gray-800">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center justify-between w-full"
              >
                <span className="flex items-center">
                  <BarChart2 className="w-5 h-5 mr-2 text-blue-400" />
                  Detailed Technical Results
                </span>
                <ChevronDown className={`w-5 h-5 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
              </button>
              
              {showAdvanced && (
                <div className="mt-4 pt-4 border-t border-gray-700">
                  <div className="mb-4">
                    <h4 className="text-sm font-semibold mb-2">Raw Prediction Output</h4>
                    <div className="p-2 font-mono text-sm bg-gray-900 rounded">
                      {formMetadata.problem_type === 'regression' ? (
                        `Predicted Value: ${predictionResult.prediction}`
                      ) : (
                        `Predicted Class: ${predictionResult.prediction}`
                      )}
                    </div>
                  </div>
                  
                  {showProbs && predictionResult.probabilities && (
                    <div>
                      <h4 className="text-sm font-semibold mb-2">Prediction Probabilities</h4>
                      {formMetadata.problem_type === 'binary_classification' ? (
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-3 rounded-md bg-gray-700">
                            <div className="text-sm text-gray-400">Probability (Class 0/Negative)</div>
                            <div className="text-xl font-bold">
                              {((1 - predictionResult.probabilities) * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div className="p-3 rounded-md bg-gray-700">
                            <div className="text-sm text-gray-400">Probability (Class 1/Positive)</div>
                            <div className="text-xl font-bold">
                              {(predictionResult.probabilities * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-3">
                          {predictionResult.probabilities.map((prob, idx) => (
                            <div key={idx} className="p-3 rounded-md bg-gray-700">
                              <div className="text-sm text-gray-400">Class {idx} Probability</div>
                              <div className="text-xl font-bold">
                                {(prob * 100).toFixed(1)}%
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
            
            {/* Input Summary */}
            <div className="p-3 rounded-md bg-gray-800">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center justify-between w-full"
              >
                <span className="flex items-center">
                  <Circle className="w-5 h-5 mr-2 text-blue-400" />
                  Input Summary
                </span>
                <ChevronDown className={`w-5 h-5 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
              </button>
              
              {showAdvanced && (
                <div className="mt-4 pt-4 border-t border-gray-700">
                  <div className="grid grid-cols-2 gap-4">
                    {Object.entries(predictionResult.input_data).map(([key, value]) => (
                      <div key={key} className="p-2 rounded-md bg-gray-700">
                        <div className="text-sm text-gray-400">
                          {key.replace('_', ' ').title()}
                        </div>
                        <div className="font-medium">{value}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            
            {/* Done Button */}
            <div className="flex justify-center pt-4">
              <button
                onClick={() => {
                  setPredictionResult(null);
                  onClose();
                }}
                className="px-6 py-2 font-medium rounded-md bg-blue-600 hover:bg-blue-700 transition-colors"
              >
                Done
              </button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit(onSubmit)}>
            {/* Input Features */}
            <div className="mb-6">
              <h3 className="flex items-center mb-4 text-lg font-semibold">
                <Crosshair className="w-5 h-5 mr-2 text-yellow-400" />
                Input Features
              </h3>
              
              <div className="space-y-6">
                {Object.entries(formMetadata.form_fields).map(([fieldName, fieldConfig]) => {
                  const displayName = fieldName.replace('_', ' ').title();
                  const inputType = fieldConfig.input_type;
                  
                  return (
                    <div key={fieldName} className="space-y-2">
                      <label className="block font-medium">{displayName}</label>
                      
                      {fieldConfig.description && (
                        <p className="text-sm text-gray-400">{fieldConfig.description}</p>
                      )}
                      
                      {inputType === 'number' ? (
                        <div className="space-y-2">
                          {fieldConfig.min !== undefined && fieldConfig.max !== undefined && (
                            <div className="flex items-center justify-between text-sm text-gray-400">
                              <span>Min: {fieldConfig.min}</span>
                              <span>Max: {fieldConfig.max}</span>
                            </div>
                          )}
                          
                          <input
                            {...register(fieldName)}
                            type="number"
                            min={fieldConfig.min}
                            max={fieldConfig.max}
                            step={fieldConfig.step || 1}
                            className="w-full p-2 rounded-md bg-gray-800 border border-gray-700 focus:border-blue-500 focus:outline-none"
                          />
                        </div>
                      ) : inputType === 'select' ? (
                        <select
                          {...register(fieldName)}
                          className="w-full p-2 rounded-md bg-gray-800 border border-gray-700 focus:border-blue-500 focus:outline-none"
                        >
                          {fieldConfig.options.map((option) => (
                            <option key={option} value={option}>{option}</option>
                          ))}
                        </select>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </div>
            
            {/* Prediction Options */}
            <div className="mb-6">
              <h3 className="flex items-center mb-4 text-lg font-semibold">
                <Gauge className="w-5 h-5 mr-2 text-purple-400" />
                Prediction Options
              </h3>
              
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showProbs}
                  onChange={(e) => setShowProbs(e.target.checked)}
                  disabled={formMetadata.problem_type === 'regression'}
                  className="w-4 h-4 mr-2 rounded bg-gray-800 border-gray-700 focus:ring-blue-500"
                />
                <span className="text-sm">Show Probabilities</span>
              </label>
              
              {formMetadata.problem_type === 'regression' && (
                <p className="mt-1 text-xs text-gray-400">
                  (Only available for classification problems)
                </p>
              )}
            </div>
            
            {/* Form Buttons */}
            <div className="grid grid-cols-3 gap-4">
              <button
                type="submit"
                disabled={isPredicting}
                className="flex items-center justify-center px-4 py-2 font-medium rounded-md bg-green-600 hover:bg-green-700 disabled:bg-gray-600 transition-colors"
              >
                <Rocket className="w-4 h-4 mr-2" />
                Make Prediction
              </button>
              <button
                type="button"
                onClick={handleReset}
                disabled={isPredicting}
                className="flex items-center justify-center px-4 py-2 font-medium rounded-md bg-gray-700 hover:bg-gray-600 disabled:bg-gray-600 transition-colors"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Reset Form
              </button>
              <button
                type="button"
                onClick={onClose}
                disabled={isPredicting}
                className="flex items-center justify-center px-4 py-2 font-medium rounded-md bg-red-600 hover:bg-red-700 disabled:bg-gray-600 transition-colors"
              >
                <X className="w-4 h-4 mr-2" />
                Cancel
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default PredictionForm;