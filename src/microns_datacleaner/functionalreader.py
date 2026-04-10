# ----- V1 of the reader class for the H5 file by Dario Liscai
# Some modifications + comments by VB. Active development, expect breaking changes

from pathlib import Path
import h5py
import numpy as np

class MicronsFunctionalReader:
    """
    The main class to work with the simplified version of the functional data
    """

    """
    Path where the code is executed
    """
    homedir = Path().resolve() 

    """
    Subfolder aimed to contain the downloaded data
    """
    filename = "functional/microns_functional.h5" 

    def __init__(self, datadir="data", path=None):
        """
        Open the functional data. 

        Parameters
        ----------
            datadir : str,
                defaults to 'data'; the folder where the data is stored. Should coincide with the one in the DataCleaner class. 
                If the functional data was downloaded by the the package, specifying the datadir is enough to find the file. 
            
            path : str,
                defaults to None. This argument should point to the .h5 file directly. When specified, it overrides the datadir argument. 

        """

        if path is not None:
            self.file_path = path
        else:
            self.file_path = f"{self.homedir}/{datadir}/{self.filename}"
        
        self.f = h5py.File(self.file_path, 'r')

    def close(self):
        """
        Close the file
        """
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def _encode_hash(self, h):
        """
        Encodes a hash in a correct format for files

        Parameters
        ----------
            h : str,
                Hash to be encoded 
        """
        return h.replace('/', '%2F')

    def _decode_hash(self, h):
        """
        Decodes the hash 


        Parameters
        ----------
            h : str,
                Hash to be decoded
        """
        return h.replace('%2F', '/')


    def _read_trial(self, trial_grp, session_key, brain_area=None):
        """
        Helper to extract all datasets from a trial group.
        Centralizes field names so they only need updating in one place.

        Parameters:
        -----------
            trial_grp : dict
                Dictionary containing all the trials
            session_key : str
                The current session_scan pair, e.g. '4_7'
            brain_area : str
                The selected brain area, e.g. 'V1'

        Return
        ----------
            trial_data : dict
                A dictionary with all trial data, including the 'session', 'scan_idx', and the 
                neural and behavioural responses happening at 'stim_times'
        """
        if brain_area:
            area_path = f"sessions/{session_key}/meta/area_indices/{brain_area}"
            if area_path not in self.f:
                return None
            indices = self.f[area_path][:]
            responses = trial_grp['responses'][indices, :]
        else:
            responses = trial_grp['responses'][:]

        return {
            'session':    session_key,
            'trial_idx':  trial_grp.name.split('/')[-1],
            'responses':  responses,
            'treadmill':  trial_grp['treadmill'][:],
            'pupil':      trial_grp['pupil'][:],
            'stim_times': trial_grp['stim_times'][:],
        }

    def get_full_data_by_hash(self, condition_hash, brain_area=None):
        """
        Returns a dictionary with the clip and all trials (responses, treadmill,
        pupil, stim_times) associated with a condition hash.

        Parameters:
        -----------
            condition_hash : str
                The identifier for the video.
            brain_area : str, optional 
                Filter responses only in this area 

        Returns:
        --------
            hash_data : dict
                Contains the 'clip' with all video frames, a 'stim_type' being Monet2, Trippy or Clip, and
                'trials', an array of dictionaries with trial data.
        """

        h_key = self._encode_hash(condition_hash)
        vdata = self.get_video_data(condition_hash)
        clip, stim_type = vdata['clip'], vdata['stim_type']
        if clip is None:
            return None

        data_out = {'clip': clip, 'stim_type': stim_type, 'trials': []}

        instances = self.f[f'videos/{h_key}/instances']
        for instance_name in instances:
            trial_grp = instances[instance_name]
            session_key = "_".join(instance_name.split('_')[:2])
            trial = self._read_trial(trial_grp, session_key, brain_area)
            if trial is not None:
                data_out['trials'].append(trial)

        return data_out

    def get_responses_by_hash(self, condition_hash, brain_area=None):
        """
        Returns all neural responses associated with a condition hash.

        Parameters:
        -----------
            condition_hash : str
                The identifier for the video.
            brain_area : str, optional 
                Filter responses only in this area 

        Returns:
        --------
            responses: dict
                A dictionary including the session and trial idx, as well as the responses themselves
        """

        full_data = self.get_full_data_by_hash(condition_hash, brain_area=brain_area)
        if full_data is None:
            return []
        return [
            {
                'session':   t['session'],
                'trial_idx': t['trial_idx'],
                'responses': t['responses'],
            }
            for t in full_data['trials']
        ]

    def get_video_data(self, condition_hash):
        """
        Returns the video data for a hash 

        Parameters:
        -----------
            condition_hash : str
                The identifier for the video.

        Returns:
        --------
            video_data : dict 
                Dictionary with all the properties of the requested video. `clip` contains frames in FxWxH format and `stim_type` the stimulus type. 
                For Monet2 trials, the `directions` and `onsets` are also available. For Clip trials, `movie_name` and short name are included. 
                Other constants of the videos are also available.  
        """
        h_key = self._encode_hash(condition_hash)
        video_path = f"videos/{h_key}"
        if video_path not in self.f:
            return None, None
        vid_grp = self.f[video_path]
        result = {}
        result['clip']      = vid_grp['clip'][:]
        result['stim_type'] =  vid_grp.attrs.get('type', 'Unknown')

        match result['stim_type']:
            case "Clip":
                result['movie_name']       = vid_grp.attrs.get("movie_name")
                result['short_movie_name'] = vid_grp.attrs.get("short_movie_name")
            case "Monet2":
                result['directions']       = vid_grp["directions"][:]
                result['onsets']           = vid_grp["onsets"][:]
                result['ori_coherence']    = vid_grp.attrs.get("ori_coherence")
            case "Trippy":
                result['temp_freq']        = vid_grp.attrs.get("temp_freq")
                result['spatial_freq']     = vid_grp.attrs.get("spatial_freq")

        result['duration']         = vid_grp.attrs.get("duration")
        result['fps']              = vid_grp.attrs.get("fps")

        return result

    def get_video_type(self, condition_hash):
        """
        Returns stimulation type of a given hash 

        Parameters:
        -----------
            condition_hash : str
                The identifier for the video.

        Returns:
        --------
            stim_type : str
                Type of stimulation: Monet2, Trippy or Clip
        """

        h_key = self._encode_hash(condition_hash)
        video_path = f"videos/{h_key}"
        if video_path not in self.f:
            return None
        vid_grp = self.f[video_path]
        return vid_grp.attrs.get('type', 'Unknown')

    def get_hashes_by_session(self, session_key, return_unique=False):
        """
        Returns condition hashes shown in a specific session.

        Parameters:
        -----------
            session_key : str
                The identifier for the session and scan, e.g. '9_4'
            return_unique : str, optional
                if False (default), returns all the hashes corresponding to each session trial, even if 
                they are repeated. Same video (with same hash) can be shown more than once per session

        Returns:
        --------
            hashes : list or set 
                A list of hashes. If return_unique = True, a set is returned
        """
        if session_key not in self.f['sessions']:
            raise ValueError(f"Session {session_key} not found.")
        hashes = self.f[f'sessions/{session_key}/meta/condition_hashes'][:]
        decoded = [self._decode_hash(h.decode('utf-8')) for h in hashes]
        return set(decoded) if return_unique else decoded

    def get_hashes_by_type(self, stim_type):
        """
        Returns hashes belonging to a specific stimulus type (e.g., 'Monet2').

        Parameters:
        -----------
            stim_type : str
                The type of stimuli to filter for: Monet2, Clip or Trippy 

        Returns:
        --------
            hashes : list 
                A list of hashes. 
        """

        if stim_type not in self.f['types']:
            return []
        return [self._decode_hash(k) for k in self.f[f'types/{stim_type}'].keys()]

    def get_available_brain_areas(self, session_key=None):
        """
        Returns brain areas available in the file or a specific session.

        Parameters:
        -----------
            session_key : str, optional
                The identifier for the session and scan, e.g. '9_4'

        Returns:
        --------
            areas : list 
                A list with the identifiers of the brain areas present.
        """

        if session_key:
            return list(self.f[f'sessions/{session_key}/meta/area_indices'].keys())
        return list(self.f['brain_areas'].keys())

    def get_trial(self, session_key, trial_idx, brain_area=None):
        """
        Direct access to a single trial by session and trial index.

        Parameters:
        -----------
            session_key : str, optional
                The identifier for the session and scan, e.g. '9_4'
            trial_idx : int or str 
                The index of the desired trial
            brain_area : str, optional 
                Filter responses only in this area 

        Returns:
        --------
            trial_data : dict
                A dictionary with all trial data, including the 'session', 'scan_idx', and the 
                neural and behavioural responses happening at 'stim_times'

        """
        trial_path = f"sessions/{session_key}/trials/{trial_idx}"
        if trial_path not in self.f:
            raise ValueError(f"Trial {trial_idx} not found in session {session_key}.")
        return self._read_trial(self.f[trial_path], session_key, brain_area)

    def print_structure(self, max_items=5, follow_links=False):
        """
        Print a tree-like structure of the data file

        Parameters:
        -----------
            max_items: int, optional 
                Number of items to show in the tree at most, defaults to 5
            follow_links: bool 
                Show the contents of internal links in the file. Defaults to False.
        Returns:
        --------
            None
        """
        print(f"\nStructure of: {self.file_path}")
        print("=" * 50)

        def _print_tree(name, obj, indent="", current_key=""):
            item_name = current_key if current_key else name
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📄 {item_name:20} [Dataset: {obj.shape}, {obj.dtype}]")
                return
            attrs = dict(obj.attrs)
            attr_str = f"  | Attributes: {attrs}" if attrs else ""
            print(f"{indent}📂 {item_name.upper()}/ {attr_str}")
            keys = sorted(obj.keys())
            for key in keys[:max_items]:
                link_obj = obj.get(key, getlink=True)
                if isinstance(link_obj, h5py.SoftLink):
                    if follow_links:
                        _print_tree(key, obj[key], indent + "    ", current_key=key)
                    else:
                        print(f"{indent}    🔗 {key:18} -> {link_obj.path}")
                else:
                    _print_tree(key, obj[key], indent + "    ", current_key=key)
            if len(keys) > max_items:
                print(f"{indent}    ... and {len(keys) - max_items} more items")

        for key in sorted(self.f.keys()):
            _print_tree(key, self.f[key], current_key=key)
