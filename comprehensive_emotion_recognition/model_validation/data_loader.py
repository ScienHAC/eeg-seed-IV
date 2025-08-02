"""
Load and process unseen SEED-IV data for model validation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from data_processing.seed_iv_loader import SeedIVLoader
    # from data_processing.feature_engineering import FeatureEngineer
except ImportError as e:
    logging.error(f"Import error: {e}")

logger = logging.getLogger(__name__)

class UnseenDataLoader:
    """
    Loads and processes unseen SEED-IV data for validation
    """
    
    def __init__(self, config):
        self.config = config
        # Create a data config for the loader
        self.data_config = self._create_data_config()
        
    def _create_data_config(self):
        """Create data configuration for the loader"""
        class DataConfig:
            def __init__(self, config):
                self.seed_iv_base_path = config.data_dir
                self.csv_output_dir = config.validation_output_dir
                self.n_subjects = 15
                self.n_sessions = 3
                self.n_trials = 24
                self.n_channels = 62
                self.n_frequency_bands = 5
                self.emotions = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
                self.n_classes = 4
        
        return DataConfig(self.config)
    
    def load_csv_data_directly(self, test_subjects: List[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load CSV data directly for testing
        Structure: csv/[session]/[subject]/de_LDS[trial].csv
        
        Parameters:
        -----------
        test_subjects : List[int], optional
            List of subject IDs to use for testing
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, Dict]
            (features, labels, metadata)
        """
        if test_subjects is None:
            test_subjects = self.config.test_subjects
        
        logger.info(f"Loading unseen data from subjects: {test_subjects}")
        logger.info(f"Data directory: {self.config.data_dir}")
        
        all_features = []
        all_labels = []
        all_subjects = []
        all_sessions = []
        
        # Define emotion labels for each session (SEED-IV standard)
        session_labels = {
            1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],  # 24 trials
            2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1], 
            3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
        }
        
        csv_base = Path(self.config.data_dir)
        
        if not csv_base.exists():
            logger.error(f"Data directory does not exist: {csv_base}")
            return None, None, {}
        
        logger.info(f"Loading from CSV directory structure: {csv_base}")
        
        # Load data for each session and subject
        for session in [1, 2, 3]:
            session_dir = csv_base / str(session)
            if not session_dir.exists():
                logger.warning(f"Session {session} directory not found: {session_dir}")
                continue
                
            for subject in test_subjects:
                subject_dir = session_dir / str(subject)
                if not subject_dir.exists():
                    logger.warning(f"Subject {subject} directory not found in session {session}: {subject_dir}")
                    continue
                
                logger.info(f"Loading Subject {subject}, Session {session}")
                
                # Load all 24 trials for this subject-session
                for trial in range(1, 25):  # 1 to 24
                    csv_file = subject_dir / f"de_LDS{trial}.csv"
                    
                    if not csv_file.exists():
                        logger.warning(f"Trial file not found: {csv_file}")
                        continue
                    
                    try:
                        # Load CSV file
                        trial_data = pd.read_csv(csv_file, header=None)
                        
                        # Average across time dimension (if multiple rows)
                        if len(trial_data.shape) > 1 and trial_data.shape[0] > 1:
                            features = trial_data.mean(axis=0).values  # Average across time
                        else:
                            features = trial_data.values.flatten()  # Already flattened
                        
                        # Get label for this trial
                        trial_label = session_labels[session][trial - 1]  # trial-1 because list is 0-indexed
                        
                        all_features.append(features)
                        all_labels.append(trial_label)
                        all_subjects.append(subject)
                        all_sessions.append(session)
                        
                    except Exception as e:
                        logger.error(f"Error loading {csv_file}: {e}")
                        continue
        
        if not all_features:
            logger.error("No data loaded! Check subject directories and file structure.")
            return None, None, {}
        
        try:
            # Convert to numpy arrays
            features = np.array(all_features)
            labels = np.array(all_labels)
            subjects = np.array(all_subjects)
            sessions = np.array(all_sessions)
            
            logger.info(f"✅ Loaded {len(features)} samples from {len(np.unique(subjects))} test subjects")
            logger.info(f"✅ Feature shape: {features.shape}")
            logger.info(f"✅ Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
            
            # Create metadata
            test_metadata = {
                'n_samples': len(features),
                'n_subjects': len(np.unique(subjects)),
                'subjects': np.unique(subjects).tolist(),
                'sessions': np.unique(sessions).tolist(),
                'feature_shape': features.shape,
                'label_distribution': {
                    int(label): int(count) for label, count in 
                    zip(*np.unique(labels, return_counts=True))
                },
                'subjects_per_session': {
                    session: len(np.unique(subjects[sessions == session]))
                    for session in np.unique(sessions)
                }
            }
            
            return features, labels, test_metadata
            
        except Exception as e:
            logger.error(f"Failed to process loaded data: {e}")
            return None, None, {}
    
    def create_stratified_test_set(self, features: np.ndarray, labels: np.ndarray, 
                                 test_size: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train/test split for comparison
        
        Parameters:
        -----------
        features : np.ndarray
            Feature array
        labels : np.ndarray
            Label array
        test_size : float
            Fraction for test set
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (X_train, X_test, y_train, y_test)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, 
                test_size=test_size,
                stratify=labels,
                random_state=self.config.random_state
            )
            
            logger.info(f"Created stratified split: {len(X_train)} train, {len(X_test)} test samples")
            
            return X_train, X_test, y_train, y_test
        except ImportError:
            logger.error("sklearn not available for stratified split")
            # Simple random split
            n_test = int(len(features) * test_size)
            indices = np.random.permutation(len(features))
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
            
            return features[train_indices], features[test_indices], labels[train_indices], labels[test_indices]
    
    def get_data_statistics(self, features: np.ndarray, labels: np.ndarray, subjects: np.ndarray = None) -> Dict:
        """
        Get comprehensive statistics about the loaded data
        
        Parameters:
        -----------
        features : np.ndarray
            Feature array
        labels : np.ndarray
            Label array
        subjects : np.ndarray, optional
            Subject array
            
        Returns:
        --------
        Dict
            Dictionary with data statistics
        """
        stats = {
            'n_samples': len(features),
            'n_features': features.shape[1] if len(features.shape) > 1 else 0,
            'label_distribution': {},
            'feature_statistics': {
                'mean': np.mean(features, axis=0) if len(features) > 0 else None,
                'std': np.std(features, axis=0) if len(features) > 0 else None,
                'min': np.min(features, axis=0) if len(features) > 0 else None,
                'max': np.max(features, axis=0) if len(features) > 0 else None
            }
        }
        
        # Label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            stats['label_distribution'][int(label)] = {
                'count': int(count),
                'percentage': float(count / len(labels) * 100)
            }
        
        # Subject distribution if available
        if subjects is not None:
            unique_subjects, subject_counts = np.unique(subjects, return_counts=True)
            stats['n_subjects'] = len(unique_subjects)
            stats['subject_distribution'] = {
                int(subj): int(count) for subj, count in zip(unique_subjects, subject_counts)
            }
        
        return stats
    
    def load_unseen_test_data(self, test_subjects: List[int] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load unseen test data for validation
        First tries to load from MATLAB files, then falls back to CSV files
        
        Parameters:
        -----------
        test_subjects : List[int], optional
            List of test subjects to use. If None, uses config.test_subjects
            
        Returns:
        --------
        Optional[Tuple[np.ndarray, np.ndarray]]
            (X_test, y_test) if successful, None otherwise
        """
        try:
            if test_subjects is None:
                test_subjects = self.config.test_subjects
            
            logger.info(f"Loading unseen test data for subjects: {test_subjects}")
            
            # First try to load from original MATLAB files
            matlab_dir = getattr(self.config, 'matlab_data_dir', None)
            if matlab_dir and Path(matlab_dir).exists():
                logger.info(f"Trying to load from MATLAB directory: {matlab_dir}")
                result = self._load_from_matlab_files(test_subjects, matlab_dir)
                if result is not None:
                    return result
                else:
                    logger.info("MATLAB loading failed, falling back to CSV files")
            
            # Fallback to CSV files
            logger.info(f"Loading from CSV directory: {self.config.data_dir}")
            X_test, y_test, metadata = self.load_csv_data_directly(test_subjects)
            
            if X_test is None or y_test is None:
                logger.error("Failed to load test data from both MATLAB and CSV sources")
                return None
            
            logger.info(f"✅ Successfully loaded unseen test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            logger.info(f"✅ Label distribution: {np.bincount(y_test.astype(int))}")
            
            return X_test, y_test
            
        except Exception as e:
            logger.error(f"Error loading unseen test data: {e}")
            return None
    
    def _load_from_matlab_files(self, test_subjects: List[int], matlab_dir: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load data from original MATLAB .mat files
        
        Parameters:
        -----------
        test_subjects : List[int]
            List of test subjects
        matlab_dir : str
            Path to MATLAB files directory
            
        Returns:
        --------
        Optional[Tuple[np.ndarray, np.ndarray]]
            (X_test, y_test) if successful, None otherwise
        """
        try:
            # Try to import scipy for .mat file loading
            try:
                from scipy.io import loadmat
            except ImportError:
                logger.warning("scipy not available, cannot load .mat files")
                return None
            
            matlab_path = Path(matlab_dir)
            logger.info(f"Scanning MATLAB directory: {matlab_path}")
            
            # Look for .mat files in the directory
            mat_files = list(matlab_path.glob("*.mat"))
            if not mat_files:
                logger.warning(f"No .mat files found in {matlab_path}")
                return None
            
            logger.info(f"Found {len(mat_files)} .mat files")
            
            all_features = []
            all_labels = []
            
            # Session labels for SEED-IV
            session_labels = {
                1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
                2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1], 
                3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
            }
            
            # Try to load data for test subjects
            for subject in test_subjects:
                for session in [1, 2, 3]:
                    # Look for files matching pattern for this subject/session
                    pattern = f"*{subject}*{session}*.mat"
                    subject_files = list(matlab_path.glob(pattern))
                    
                    if not subject_files:
                        logger.warning(f"No .mat files found for subject {subject}, session {session}")
                        continue
                    
                    for mat_file in subject_files:
                        try:
                            # Load .mat file
                            mat_data = loadmat(str(mat_file))
                            
                            # Extract features (this depends on your .mat file structure)
                            # You may need to adjust these keys based on your actual .mat file structure
                            if 'de_LDS' in mat_data:
                                features = mat_data['de_LDS']
                            elif 'features' in mat_data:
                                features = mat_data['features']
                            else:
                                # Try to find the main data array
                                data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                                if data_keys:
                                    features = mat_data[data_keys[0]]
                                else:
                                    logger.warning(f"Could not find feature data in {mat_file}")
                                    continue
                            
                            # Process features (reshape if needed)
                            if features.ndim > 2:
                                features = features.reshape(features.shape[0], -1)
                            
                            # Get labels for this session
                            labels = session_labels[session]
                            
                            # Add data
                            for i, trial_features in enumerate(features):
                                if i < len(labels):
                                    all_features.append(trial_features.flatten())
                                    all_labels.append(labels[i])
                            
                            logger.info(f"Loaded {features.shape[0]} trials from {mat_file.name}")
                            
                        except Exception as e:
                            logger.error(f"Error loading {mat_file}: {e}")
                            continue
            
            if not all_features:
                logger.warning("No data loaded from MATLAB files")
                return None
            
            X_test = np.array(all_features)
            y_test = np.array(all_labels)
            
            logger.info(f"✅ Loaded {len(X_test)} samples from MATLAB files")
            logger.info(f"✅ Feature shape: {X_test.shape}")
            
            return X_test, y_test
            
        except Exception as e:
            logger.error(f"Error loading from MATLAB files: {e}")
            return None
