import React, { useState, useContext } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'react-toastify';
import { 
  Rocket, RefreshCw, X, ChevronDown, Gauge, Crosshair, 
  TrendingUp, Activity, Layers, Zap, AlertCircle, CheckCircle 
} from 'lucide-react';
import { useAppContext } from '../context/AppContext';

// Default configuration
const DEFAULT_CONFIG = {
  test_size: 0.2,
  val_size: 0.2,
  random_state: 42,
  epochs: 30,
  batch_size: 32,
  early_stopping_patience: 10,
  model_params: {
    hidden_layers: [64, 32],
    dropout_rate: 0.2,
    l2_reg: 0.01,
    learning_rate: 0.001
  }
};

const TrainModel = ({ isRetrain = false, onClose }) => {
  const { 
    selectedDataset, 
    selectedTable, 
    projectId = "bigdata-sprint",
    bucket = "my-smart-ingest-bucket",
    bq
  } = useContext(AppContext);
  
  const { register, handleSubmit, reset, watch } = useForm({
    defaultValues: {
      target_column: '',
      use_custom_config: false,
      ...DEFAULT_CONFIG,
      model_params: DEFAULT_CONFIG.model_params
    }
  });
  
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [trainingResult, setTrainingResult] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  const useCustomConfig = watch('use_custom_config');
  
  const onSubmit = async (data) => {
    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingStatus('Initializing training...');
    
    try {
      // Simulate training progress
      const simulateProgress = () => {
        return new Promise((resolve) => {
          const intervals = [10, 20, 40, 50, 70, 100];
          intervals.forEach((progress, i) => {
            setTimeout(() => {
              setTrainingProgress(progress);
              setTrainingStatus([
                'Initializing training...',
                'Loading data...',
                'Sending training request...',
                'Training the model...',
                'Processing training request...',
                'Training completed!'
              ][i]);
              if (progress === 100) resolve();
            }, i * 2000);
          });
        });
      };
      
      await simulateProgress();
      
      // Simulate API response
      setTimeout(() => {
        const mockResult = {
          status: "success",
          metrics: {
            accuracy: 0.92,
            precision: 0.89,
            recall: 0.91,
            f1: 0.90
          },
          problem_type: "binary_classification",
          target_column: data.target_column || "auto_detected",
          input_size: 15,
          num_classes: 2,
          model_path: `gs://${bucket}/models/${selectedDataset}_${selectedTable}_model.h5`
        };
        
        setTrainingResult(mockResult);
        toast.success(
          <div className="flex items-center">
            <CheckCircle className="w-5 h-5 mr-2 text-green-500" />
            Training completed successfully!
          </div>,
          { className: 'bg-gray-800 text-white' }
        );
      }, 1000);
      
    } catch (error) {
      toast.error(
        <div className="flex items-center">
          <AlertCircle className="w-5 h-5 mr-2 text-red-500" />
          Training failed: {error.message}
        </div>,
        { className: 'bg-gray-800 text-white' }
      );
    } finally {
      setIsTraining(false);
    }
  };

  const handleReset = () => {
    reset();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="w-full max-w-4xl p-6 rounded-lg shadow-xl bg-gray-900 text-white">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">
            {isRetrain ? (
              <span className="flex items-center">
                <RefreshCw className="w-5 h-5 mr-2 text-blue-400" />
                Retrain Model
              </span>
            ) : (
              <span className="flex items-center">
                <Rocket className="w-5 h-5 mr-2 text-green-400" />
                Train New Model
              </span>
            )}
          </h2>
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
              <span className="font-semibold">Project:</span> 
              <span className="ml-2 font-mono text-blue-300">{projectId}</span>
            </div>
            <div>
              <span className="font-semibold">Bucket:</span> 
              <span className="ml-2 font-mono text-blue-300">{bucket}</span>
            </div>
            <div>
              <span className="font-semibold">Dataset:</span> 
              <span className="ml-2 font-mono text-blue-300">{selectedDataset}</span>
            </div>
            <div>
              <span className="font-semibold">Table:</span> 
              <span className="ml-2 font-mono text-blue-300">{selectedTable}</span>
            </div>
          </div>
        </div>
        
        {isTraining ? (
          <div className="space-y-4">
            <h3 className="flex items-center text-lg font-semibold">
              <Activity className="w-5 h-5 mr-2 text-yellow-400" />
              Training in Progress...
            </h3>
            <p className="flex items-center text-sm text-gray-300">
              <Zap className="w-4 h-4 mr-2 text-yellow-400" />
              {trainingStatus}
            </p>
            <div className="w-full h-2 bg-gray-700 rounded-full">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 to-green-500 rounded-full transition-all duration-500"
                style={{ width: `${trainingProgress}%` }}
              ></div>
            </div>
            <div className="text-right text-sm text-gray-400">
              {trainingProgress}% complete
            </div>
          </div>
        ) : trainingResult ? (
          <div className="space-y-6">
            {/* Success Message */}
            <div className="p-4 rounded-md bg-green-900 bg-opacity-30 border border-green-800">
              <div className="flex items-center">
                <CheckCircle className="w-5 h-5 mr-2 text-green-400" />
                <span className="font-semibold">Training Completed Successfully!</span>
              </div>
            </div>
            
            {/* Metrics */}
            <div>
              <h3 className="flex items-center mb-4 text-lg font-semibold">
                <TrendingUp className="w-5 h-5 mr-2 text-purple-400" />
                Model Performance Metrics
              </h3>
              <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="p-3 rounded-md bg-gray-800">
                  <div className="text-sm text-gray-400">Accuracy</div>
                  <div className="text-2xl font-bold">
                    {trainingResult.metrics.accuracy.toFixed(4)}
                  </div>
                  <div className="text-sm text-green-400">
                    {(trainingResult.metrics.accuracy * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="p-3 rounded-md bg-gray-800">
                  <div className="text-sm text-gray-400">Precision</div>
                  <div className="text-2xl font-bold">
                    {trainingResult.metrics.precision.toFixed(4)}
                  </div>
                  <div className="text-sm text-green-400">
                    {(trainingResult.metrics.precision * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="p-3 rounded-md bg-gray-800">
                  <div className="text-sm text-gray-400">Recall</div>
                  <div className="text-2xl font-bold">
                    {trainingResult.metrics.recall.toFixed(4)}
                  </div>
                  <div className="text-sm text-green-400">
                    {(trainingResult.metrics.recall * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="p-3 rounded-md bg-gray-800">
                  <div className="text-sm text-gray-400">F1 Score</div>
                  <div className="text-2xl font-bold">
                    {trainingResult.metrics.f1.toFixed(4)}
                  </div>
                  <div className="text-sm text-green-400">
                    {(trainingResult.metrics.f1 * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>
            
            {/* Model Details */}
            <div>
              <h3 className="flex items-center mb-4 text-lg font-semibold">
                <Layers className="w-5 h-5 mr-2 text-blue-400" />
                Model Details
              </h3>
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="p-3 rounded-md bg-gray-800">
                  <div className="text-sm text-gray-400">Problem Type</div>
                  <div className="text-xl font-bold capitalize">
                    {trainingResult.problem_type.replace('_', ' ')}
                  </div>
                </div>
                <div className="p-3 rounded-md bg-gray-800">
                  <div className="text-sm text-gray-400">Target Column</div>
                  <div className="text-xl font-bold">
                    {trainingResult.target_column}
                  </div>
                </div>
                <div className="p-3 rounded-md bg-gray-800">
                  <div className="text-sm text-gray-400">Input Features</div>
                  <div className="text-xl font-bold">
                    {trainingResult.input_size}
                  </div>
                </div>
                {trainingResult.num_classes && (
                  <div className="p-3 rounded-md bg-gray-800">
                    <div className="text-sm text-gray-400">Classes</div>
                    <div className="text-xl font-bold">
                      {trainingResult.num_classes}
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            {/* Model Path */}
            <div className="p-3 rounded-md bg-gray-800">
              <div className="text-sm text-gray-400">Model Path</div>
              <div className="font-mono text-blue-300 break-all">
                {trainingResult.model_path}
              </div>
            </div>
            
            {/* Done Button */}
            <div className="flex justify-center pt-4">
              <button
                onClick={() => {
                  setTrainingResult(null);
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
            {/* Target Column */}
            <div className="mb-6">
              <h3 className="flex items-center mb-2 text-lg font-semibold">
                <Crosshair className="w-5 h-5 mr-2 text-yellow-400" />
                Target Column
              </h3>
              <input
                {...register('target_column')}
                type="text"
                placeholder="Leave empty for auto-detection"
                className="w-full p-2 rounded-md bg-gray-800 border border-gray-700 focus:border-blue-500 focus:outline-none"
              />
              <p className="mt-1 text-sm text-gray-400">
                Specify the target column name. If left empty, the system will auto-detect.
              </p>
            </div>
            
            {/* Configuration Section */}
            <div className="mb-6">
              <h3 className="flex items-center mb-2 text-lg font-semibold">
                <Gauge className="w-5 h-5 mr-2 text-purple-400" />
                Training Configuration
              </h3>
              
              {/* Advanced Config Toggle */}
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center px-4 py-2 mb-4 rounded-md bg-gray-800 hover:bg-gray-700 transition-colors"
              >
                <span>🔧 Advanced Configuration</span>
                <ChevronDown className={`w-4 h-4 ml-2 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
              </button>
              
              {showAdvanced && (
                <div className="p-4 rounded-md bg-gray-800">
                  <h4 className="mb-3 font-semibold">Training Parameters</h4>
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div>
                      <label className="block mb-1 text-sm text-gray-300">Test Size</label>
                      <input
                        {...register('test_size')}
                        type="range"
                        min="0.1"
                        max="0.4"
                        step="0.05"
                        className="w-full"
                      />
                      <div className="text-sm text-gray-400">
                        Value: {watch('test_size')}
                      </div>
                    </div>
                    <div>
                      <label className="block mb-1 text-sm text-gray-300">Validation Size</label>
                      <input
                        {...register('val_size')}
                        type="range"
                        min="0.1"
                        max="0.4"
                        step="0.05"
                        className="w-full"
                      />
                      <div className="text-sm text-gray-400">
                        Value: {watch('val_size')}
                      </div>
                    </div>
                    <div>
                      <label className="block mb-1 text-sm text-gray-300">Random State</label>
                      <input
                        {...register('random_state')}
                        type="number"
                        min="1"
                        className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block mb-1 text-sm text-gray-300">Epochs</label>
                      <input
                        {...register('epochs')}
                        type="number"
                        min="1"
                        max="200"
                        className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block mb-1 text-sm text-gray-300">Batch Size</label>
                      <select
                        {...register('batch_size')}
                        className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 focus:border-blue-500 focus:outline-none"
                      >
                        <option value="16">16</option>
                        <option value="32">32</option>
                        <option value="64">64</option>
                        <option value="128">128</option>
                      </select>
                    </div>
                    <div>
                      <label className="block mb-1 text-sm text-gray-300">Early Stopping Patience</label>
                      <input
                        {...register('early_stopping_patience')}
                        type="number"
                        min="1"
                        max="50"
                        className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                  </div>
                  
                  <h4 className="mb-3 font-semibold">Model Parameters</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block mb-1 text-sm text-gray-300">Layer 1 Size</label>
                      <input
                        {...register('layer1')}
                        type="number"
                        min="8"
                        max="512"
                        step="8"
                        className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block mb-1 text-sm text-gray-300">Layer 2 Size</label>
                      <input
                        {...register('layer2')}
                        type="number"
                        min="8"
                        max="512"
                        step="8"
                        className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block mb-1 text-sm text-gray-300">Dropout Rate</label>
                      <input
                        {...register('dropout_rate')}
                        type="range"
                        min="0"
                        max="0.5"
                        step="0.05"
                        className="w-full"
                      />
                      <div className="text-sm text-gray-400">
                        Value: {watch('dropout_rate')}
                      </div>
                    </div>
                    <div>
                      <label className="block mb-1 text-sm text-gray-300">L2 Regularization</label>
                      <input
                        {...register('l2_reg')}
                        type="number"
                        min="0"
                        max="0.1"
                        step="0.001"
                        className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block mb-1 text-sm text-gray-300">Learning Rate</label>
                      <input
                        {...register('learning_rate')}
                        type="number"
                        min="0.0001"
                        max="0.01"
                        step="0.0001"
                        className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                    <div className="flex items-end">
                      <label className="flex items-center">
                        <input
                          {...register('use_custom_config')}
                          type="checkbox"
                          className="w-4 h-4 mr-2 rounded bg-gray-700 border-gray-600 focus:ring-blue-500"
                        />
                        <span className="text-sm">Use Custom Configuration</span>
                      </label>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            {/* Form Buttons */}
            <div className="grid grid-cols-3 gap-4">
              <button
                type="submit"
                disabled={isTraining}
                className="flex items-center justify-center px-4 py-2 font-medium rounded-md bg-green-600 hover:bg-green-700 disabled:bg-gray-600 transition-colors"
              >
                <Rocket className="w-4 h-4 mr-2" />
                Start Training
              </button>
              <button
                type="button"
                onClick={handleReset}
                disabled={isTraining}
                className="flex items-center justify-center px-4 py-2 font-medium rounded-md bg-gray-700 hover:bg-gray-600 disabled:bg-gray-600 transition-colors"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Reset Form
              </button>
              <button
                type="button"
                onClick={onClose}
                disabled={isTraining}
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

export default TrainModel;