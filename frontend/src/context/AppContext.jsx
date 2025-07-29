// Example Usage:
// import { useAppContext } from '../context/AppContext';
// const { currentPage, navigateTo, bq, gcs, bucket } = useAppContext();

import { createContext, useContext, useState, useMemo } from 'react';
import { BigQuery } from '@google-cloud/bigquery';
import { Storage } from '@google-cloud/storage';

// Create context
const AppContext = createContext();

// Context Provider
export const AppProvider = ({ children }) => {
  // 🔹 Navigation State
  const [currentPage, setCurrentPage] = useState('Dataset_Selection');
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedTable, setSelectedTable] = useState(null);
  const [mode, setMode] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);

  const navigateTo = (page) => setCurrentPage(page);
  const flowMap = {
    Dataset_Selection: null,
    Table_Selection: 'Dataset_Selection',
    File_Upload: 'Table_Selection',
    Prediction: 'Table_Selection',
  };
  const getPreviousPage = () => flowMap[currentPage] || 'Dataset_Selection';

  // 🔹 GCP Clients (singleton-style)
  const gcpClients = useMemo(() => {
    try {
      const credentials = {
        type: import.meta.env.VITE_GCP_TYPE,
        project_id: import.meta.env.VITE_GCP_PROJECT_ID,
        private_key: import.meta.env.VITE_GCP_PRIVATE_KEY?.replace(/\\n/g, '\n'),
        client_email: import.meta.env.VITE_GCP_CLIENT_EMAIL,
        token_uri: import.meta.env.VITE_GCP_TOKEN_URI,
      };

      return {
        bq: new BigQuery({ credentials }),
        gcs: new Storage({ credentials }),
        bucket: import.meta.env.VITE_GCP_BUCKET_NAME || '',
      };
    } catch (err) {
      console.error('❌ GCP client init failed:', err);
      return {};
    }
  }, []); // Only create once

  return (
    <AppContext.Provider
      value={{
        // navigation
        currentPage,
        navigateTo,
        getPreviousPage,
        selectedDataset,
        setSelectedDataset,
        selectedTable,
        setSelectedTable,
        mode,
        setMode,
        selectedModel,
        setSelectedModel,

        // gcp
        ...gcpClients,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

// Hook
export const useAppContext = () => useContext(AppContext);