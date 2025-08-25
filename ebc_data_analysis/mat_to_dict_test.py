import h5py
import scipy.io as sio
import numpy as np
from typing import Any, Dict, Union, List
import warnings


class MATFileReader:
    """Enhanced MATLAB file reader with better error handling and structure visualization."""
    
    def __init__(self):
        self.conversion_stats = {'structs_converted': 0, 'arrays_converted': 0}
    
    def load_mat_file(self, file_path: str, version_hint: str = 'auto') -> Dict[str, Any]:
        """
        Load MATLAB file with automatic version detection and fallback.
        
        Args:
            file_path: Path to the .mat file
            version_hint: 'auto', 'v7.3', or 'legacy' to guide loading strategy
            
        Returns:
            Dictionary containing the loaded data
        """
        if version_hint == 'v7.3':
            return self._load_hdf5_mat(file_path)
        elif version_hint == 'legacy':
            return self._load_legacy_mat(file_path)
        else:
            # Auto-detect version
            try:
                return self._load_legacy_mat(file_path)
            except (NotImplementedError, ValueError) as e:
                if "HDF5" in str(e) or "v7.3" in str(e):
                    warnings.warn(f"Falling back to HDF5 reader: {e}")
                    return self._load_hdf5_mat(file_path)
                else:
                    raise e
    
    def _load_legacy_mat(self, file_path: str) -> Dict[str, Any]:
        """Load MATLAB files version ≤ 7.2 using scipy.io."""
        try:
            mat_data = sio.loadmat(
                file_path, 
                struct_as_record=False, 
                squeeze_me=True,
                mat_dtype=True  # Preserve MATLAB data types
            )
            return self._convert_matlab_structs(mat_data)
        except Exception as e:
            raise ValueError(f"Failed to load legacy MAT file: {e}")
    
    def _load_hdf5_mat(self, file_path: str) -> Dict[str, Any]:
        """Load MATLAB files version 7.3+ using h5py."""
        try:
            with h5py.File(file_path, 'r') as f:
                return self._convert_hdf5_group(f)
        except Exception as e:
            raise ValueError(f"Failed to load HDF5 MAT file: {e}")
    
    def _convert_matlab_structs(self, data: Any) -> Any:
        """Recursively convert MATLAB structures to Python dictionaries."""
        if isinstance(data, dict):
            # Skip MATLAB metadata keys
            filtered_data = {k: v for k, v in data.items() 
                           if not k.startswith('__') and not k.endswith('__')}
            return {k: self._convert_matlab_structs(v) for k, v in filtered_data.items()}
        
        elif isinstance(data, sio.matlab.mat_struct):
            self.conversion_stats['structs_converted'] += 1
            result = {}
            for field in data._fieldnames:
                try:
                    field_data = getattr(data, field)
                    result[field] = self._convert_matlab_structs(field_data)
                except AttributeError:
                    result[field] = None
            return result
        
        elif isinstance(data, np.ndarray):
            return self._convert_numpy_array(data)
        
        else:
            return data
    
    def _convert_numpy_array(self, arr: np.ndarray) -> Any:
        """Convert numpy arrays with special handling for different cases."""
        self.conversion_stats['arrays_converted'] += 1
        
        # Handle scalar arrays
        if arr.ndim == 0:
            item = arr.item()
            return self._convert_matlab_structs(item) if hasattr(item, '_fieldnames') else item
        
        # Handle empty arrays
        if arr.size == 0:
            return []
        
        # Handle 1D arrays
        if arr.ndim == 1:
            # Check if it's a cell array of structs
            if len(arr) > 0 and hasattr(arr[0], '_fieldnames'):
                return [self._convert_matlab_structs(item) for item in arr]
            # Regular array
            return arr.tolist()
        
        # Handle multi-dimensional arrays
        try:
            # If array contains structs, convert recursively
            if arr.dtype == 'object':
                return [[self._convert_matlab_structs(item) for item in row] for row in arr]
            else:
                return arr.tolist()
        except:
            # Fallback for complex arrays
            return arr.tolist()
    
    def _convert_hdf5_group(self, group: h5py.Group) -> Dict[str, Any]:
        """Convert HDF5 group to dictionary (for MATLAB v7.3+ files)."""
        result = {}
        for key in group.keys():
            if key.startswith('#'):  # Skip HDF5 metadata
                continue
            item = group[key]
            if isinstance(item, h5py.Group):
                result[key] = self._convert_hdf5_group(item)
            elif isinstance(item, h5py.Dataset):
                result[key] = self._convert_hdf5_dataset(item)
        return result
    
    def _convert_hdf5_dataset(self, dataset: h5py.Dataset) -> Any:
        """Convert HDF5 dataset to appropriate Python type."""
        data = dataset[()]
        
        # Handle string arrays
        if dataset.dtype.kind in ['S', 'U']:  # byte or unicode strings
            if data.shape == ():
                return data.decode('utf-8') if isinstance(data, bytes) else str(data)
            else:
                return [item.decode('utf-8') if isinstance(item, bytes) else str(item) 
                       for item in data.flat]
        
        # Handle numeric data
        if data.shape == ():
            return data.item()
        else:
            return data.tolist()
    
    def print_structure(self, data: Dict[str, Any], max_depth: int = 3, 
                       current_depth: int = 0, prefix: str = "") -> None:
        """
        Print the hierarchical structure of the loaded data.
        
        Args:
            data: Dictionary to analyze
            max_depth: Maximum depth to traverse
            current_depth: Current traversal depth (internal use)
            prefix: String prefix for indentation (internal use)
        """
        if current_depth >= max_depth:
            return
        
        for key, value in data.items():
            if key.startswith('__'):  # Skip MATLAB metadata
                continue
                
            indent = "  " * current_depth + prefix
            
            if isinstance(value, dict):
                print(f"{indent}{key}/ (dict, {len(value)} keys)")
                if current_depth < max_depth - 1:
                    self.print_structure(value, max_depth, current_depth + 1, "")
            
            elif isinstance(value, list):
                if len(value) > 0:
                    first_item = value[0]
                    if isinstance(first_item, dict):
                        print(f"{indent}{key}[] (list of {len(value)} dicts, keys: {list(first_item.keys())[:5]}...)")
                    else:
                        print(f"{indent}{key}[] (list of {len(value)} {type(first_item).__name__})")
                else:
                    print(f"{indent}{key}[] (empty list)")
            
            elif isinstance(value, np.ndarray):
                print(f"{indent}{key} (array: {value.shape}, dtype: {value.dtype})")
            
            else:
                value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"{indent}{key} ({type(value).__name__}): {value_str}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the conversion process."""
        return self.conversion_stats.copy()


def explore_mat_file(file_path: str, max_depth: int = 3) -> Dict[str, Any]:
    """
    Convenience function to load and explore a MATLAB file structure.
    
    Args:
        file_path: Path to the .mat file
        max_depth: Maximum depth for structure printing
        
    Returns:
        Loaded data dictionary
    """
    reader = MATFileReader()
    
    print(f"Loading MATLAB file: {file_path}")
    print("-" * 50)
    
    try:
        data = reader.load_mat_file(file_path)
        print(f"Successfully loaded file!")
        print(f"Conversion stats: {reader.get_summary()}")
        print(f"\nFile structure (max depth {max_depth}):")
        print("=" * 50)
        reader.print_structure(data, max_depth)
        return data
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return {}


# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    bpod_file = './data/beh/E5LG/raw/E5LG_EBC_V_3_18_20250501_211247.mat'
    
    # Load and explore the file
    data = explore_mat_file(bpod_file, max_depth=4)
    
    # Access specific data (example)
    if 'SessionData' in data:
        session_data = data['SessionData']
        print(f"\nSession Data Keys: {list(session_data.keys())}")
        
        if 'nTrials' in session_data:
            print(f"Number of trials: {session_data['nTrials']}")
        
        # Example: Access trial events
        if 'RawEvents' in session_data and 'Trial' in session_data['RawEvents']:
            trials = session_data['RawEvents']['Trial']
            if len(trials) > 58:
                print(f"\nTrial 58 Events: {trials[58].get('Events', 'No events found')}")
