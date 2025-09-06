"""Telangana-specific data validation and utilities."""

import pandas as pd
from typing import Dict, List, Set
import re


class TelanganaDataValidator:
    """Validator for Telangana-specific data."""
    
    # Official Telangana districts
    DISTRICTS = {
        "Adilabad": "AD",
        "Bhadradri Kothagudem": "BK", 
        "Hyderabad": "HY",
        "Jagtial": "JA",
        "Jangaon": "JG",
        "Jayashankar Bhupalpally": "JB",
        "Jogulamba Gadwal": "JG",
        "Kamareddy": "KM",
        "Karimnagar": "KR",
        "Khammam": "KH",
        "Komaram Bheem Asifabad": "KB",
        "Mahabubabad": "MB",
        "Mahabubnagar": "MN",
        "Mancherial": "MC",
        "Medak": "MD",
        "Medchal-Malkajgiri": "MM",
        "Mulugu": "MU",
        "Nagarkurnool": "NK",
        "Nalgonda": "NL",
        "Narayanpet": "NP",
        "Nirmal": "NR",
        "Nizamabad": "NZ",
        "Peddapalli": "PD",
        "Rajanna Sircilla": "RS",
        "Rangareddy": "RR",
        "Sangareddy": "SG",
        "Siddipet": "SP",
        "Suryapet": "SY",
        "Vikarabad": "VB",
        "Wanaparthy": "WP",
        "Warangal Urban": "WU",
        "Warangal Rural": "WR",
        "Yadadri Bhuvanagiri": "YB"
    }
    
    # Major cities in Telangana
    MAJOR_CITIES = [
        "Hyderabad", "Warangal", "Nizamabad", "Khammam", "Karimnagar",
        "Ramagundam", "Mahbubnagar", "Nalgonda", "Adilabad", "Suryapet",
        "Miryalaguda", "Tadepalligudem", "Kothagudem", "Mancherial",
        "Sangareddy", "Siddipet", "Jagtial", "Korutla", "Mahabubabad"
    ]
    
    # Administrative divisions
    DIVISIONS = [
        "Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam"
    ]
    
    @classmethod
    def validate_district_names(cls, df: pd.DataFrame, 
                               district_col: str = 'district') -> Dict[str, List[str]]:
        """Validate district names in the dataset."""
        issues = {
            "invalid_districts": [],
            "suggested_corrections": {},
            "missing_districts": []
        }
        
        if district_col not in df.columns:
            return {"error": f"Column '{district_col}' not found in dataset"}
        
        # Get unique district names
        district_names = df[district_col].dropna().unique()
        
        for district in district_names:
            district_str = str(district).strip()
            
            # Check if district is valid
            if district_str not in cls.DISTRICTS:
                issues["invalid_districts"].append(district_str)
                
                # Suggest corrections
                suggestions = cls._suggest_district_correction(district_str)
                if suggestions:
                    issues["suggested_corrections"][district_str] = suggestions
        
        return issues
    
    @classmethod
    def _suggest_district_correction(cls, invalid_district: str) -> List[str]:
        """Suggest corrections for invalid district names."""
        suggestions = []
        
        # Simple fuzzy matching
        for valid_district in cls.DISTRICTS.keys():
            # Check for partial matches
            if invalid_district.lower() in valid_district.lower():
                suggestions.append(valid_district)
            elif valid_district.lower() in invalid_district.lower():
                suggestions.append(valid_district)
        
        # Check for common misspellings
        common_misspellings = {
            "mahabubnagar": "Mahabubnagar",
            "nizamabad": "Nizamabad", 
            "karimnagar": "Karimnagar",
            "warangal": "Warangal Urban",
            "hyderabad": "Hyderabad"
        }
        
        if invalid_district.lower() in common_misspellings:
            suggestions.append(common_misspellings[invalid_district.lower()])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    @classmethod
    def validate_administrative_codes(cls, df: pd.DataFrame,
                                    code_col: str = 'district_code',
                                    district_col: str = 'district') -> Dict[str, List[str]]:
        """Validate administrative codes against district names."""
        issues = {
            "invalid_codes": [],
            "mismatched_codes": []
        }
        
        if code_col not in df.columns or district_col not in df.columns:
            return {"error": "Required columns not found"}
        
        for _, row in df.iterrows():
            district = str(row[district_col]).strip()
            code = str(row[code_col]).strip()
            
            # Check if code matches district
            if district in cls.DISTRICTS:
                expected_code = cls.DISTRICTS[district]
                if code != expected_code:
                    issues["mismatched_codes"].append({
                        "district": district,
                        "actual_code": code,
                        "expected_code": expected_code
                    })
            else:
                issues["invalid_codes"].append({
                    "district": district,
                    "code": code
                })
        
        return issues
    
    @classmethod
    def validate_geographic_coordinates(cls, df: pd.DataFrame,
                                      lat_col: str = 'latitude',
                                      lon_col: str = 'longitude') -> Dict[str, List[str]]:
        """Validate geographic coordinates for Telangana."""
        issues = {
            "out_of_bounds": [],
            "invalid_coordinates": []
        }
        
        # Telangana bounds (approximate)
        TELANGANA_BOUNDS = {
            "min_lat": 15.5,
            "max_lat": 19.5,
            "min_lon": 77.0,
            "max_lon": 81.0
        }
        
        if lat_col not in df.columns or lon_col not in df.columns:
            return {"error": "Latitude/Longitude columns not found"}
        
        for idx, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                
                # Check bounds
                if not (TELANGANA_BOUNDS["min_lat"] <= lat <= TELANGANA_BOUNDS["max_lat"]):
                    issues["out_of_bounds"].append({
                        "index": idx,
                        "latitude": lat,
                        "longitude": lon,
                        "issue": "Latitude out of Telangana bounds"
                    })
                
                if not (TELANGANA_BOUNDS["min_lon"] <= lon <= TELANGANA_BOUNDS["max_lon"]):
                    issues["out_of_bounds"].append({
                        "index": idx,
                        "latitude": lat,
                        "longitude": lon,
                        "issue": "Longitude out of Telangana bounds"
                    })
                    
            except (ValueError, TypeError):
                issues["invalid_coordinates"].append({
                    "index": idx,
                    "latitude": row[lat_col],
                    "longitude": row[lon_col]
                })
        
        return issues
    
    @classmethod
    def get_district_hierarchy(cls) -> Dict[str, Dict]:
        """Get administrative hierarchy for Telangana."""
        return {
            "state": "Telangana",
            "districts": cls.DISTRICTS,
            "divisions": cls.DIVISIONS,
            "major_cities": cls.MAJOR_CITIES
        }
    
    @classmethod
    def standardize_district_names(cls, df: pd.DataFrame,
                                  district_col: str = 'district') -> pd.DataFrame:
        """Standardize district names to official format."""
        df_standardized = df.copy()
        
        if district_col not in df.columns:
            return df_standardized
        
        # Create mapping for standardization
        standardization_map = {}
        for district in df[district_col].unique():
            district_str = str(district).strip()
            if district_str not in cls.DISTRICTS:
                suggestions = cls._suggest_district_correction(district_str)
                if suggestions:
                    standardization_map[district_str] = suggestions[0]
        
        # Apply standardization
        df_standardized[district_col] = df_standardized[district_col].replace(standardization_map)
        
        return df_standardized
    
    @classmethod
    def add_administrative_codes(cls, df: pd.DataFrame,
                               district_col: str = 'district') -> pd.DataFrame:
        """Add administrative codes to the dataset."""
        df_with_codes = df.copy()
        
        if district_col not in df.columns:
            return df_with_codes
        
        # Add district codes
        df_with_codes['district_code'] = df_with_codes[district_col].map(cls.DISTRICTS)
        
        # Add division information
        division_mapping = {
            "Hyderabad": ["Hyderabad", "Rangareddy", "Medchal-Malkajgiri", "Vikarabad"],
            "Warangal": ["Warangal Urban", "Warangal Rural", "Mahabubabad", "Mulugu"],
            "Nizamabad": ["Nizamabad", "Kamareddy", "Mancherial"],
            "Karimnagar": ["Karimnagar", "Jagtial", "Peddapalli", "Rajanna Sircilla"],
            "Khammam": ["Khammam", "Bhadradri Kothagudem", "Jayashankar Bhupalpally"]
        }
        
        def get_division(district):
            for division, districts in division_mapping.items():
                if district in districts:
                    return division
            return "Other"
        
        df_with_codes['division'] = df_with_codes[district_col].apply(get_division)
        
        return df_with_codes

