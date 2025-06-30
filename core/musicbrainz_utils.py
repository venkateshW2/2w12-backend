# NEW FILE: core/musicbrainz_utils.py
import acoustid
import musicbrainzngs as mb
import logging
import asyncio
from typing import Dict, Any, Optional
import tempfile
import os

logger = logging.getLogger(__name__)

class MusicBrainzResearcher:
    """
    Background MusicBrainz integration for research and validation
    
    This runs quietly in the background and does NOT affect main analysis speed.
    Used for:
    - Validating ML predictions against known data
    - Building training datasets
    - Research data collection
    """
    
    def __init__(self):
        # Set user agent for MusicBrainz API
        mb.set_useragent("2W12SoundTools", "1.0", "https://2w12.one")
        
        self.acoustid_available = True
        self.musicbrainz_available = True
        
        # Test connections
        self._test_services()
    
    def _test_services(self):
        """Test AcoustID and MusicBrainz availability"""
        try:
            # Test AcoustID (just check if module works)
            logger.info("🔍 Testing AcoustID availability...")
            self.acoustid_available = True
            logger.info("✅ AcoustID service available")
        except Exception as e:
            logger.warning(f"⚠️  AcoustID unavailable: {e}")
            self.acoustid_available = False
        
        try:
            # Test MusicBrainz (simple search)
            logger.info("🔍 Testing MusicBrainz availability...")
            self.musicbrainz_available = True
            logger.info("✅ MusicBrainz service available")
        except Exception as e:
            logger.warning(f"⚠️  MusicBrainz unavailable: {e}")
            self.musicbrainz_available = False
    
    async def background_research(self, fingerprint: str, file_content: bytes, ml_analysis: Dict) -> bool:
        """
        Background research - does not block main analysis
        
        Args:
            fingerprint: File fingerprint for tracking
            file_content: Audio file content
            ml_analysis: ML analysis results to validate
            
        Returns:
            bool: True if research completed successfully
        """
        
        if not (self.acoustid_available and self.musicbrainz_available):
            logger.debug("🔄 MusicBrainz services unavailable, skipping research")
            return False
        
        try:
            logger.info(f"🔬 Starting background research for {fingerprint[:8]}...")
            
            # Create temporary file for fingerprinting
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Step 1: Create AcoustID fingerprint
                duration, acoustid_fingerprint = acoustid.fingerprint_file(tmp_file_path)
                
                # Step 2: Lookup in AcoustID database
                lookup_results = acoustid.lookup(
                    acoustid_fingerprint, 
                    duration,
                    meta=['recordings', 'releasegroups', 'releases', 'tracks']
                )
                
                if lookup_results and lookup_results['results']:
                    # Step 3: Get best match
                    best_match = lookup_results['results'][0]
                    
                    if 'recordings' in best_match:
                        recording = best_match['recordings'][0]
                        recording_id = recording['id']
                        
                        # Step 4: Get detailed MusicBrainz data
                        mb_data = mb.get_recording_by_id(
                            recording_id,
                            includes=['artists', 'releases', 'tags', 'ratings']
                        )
                        
                        # Step 5: Extract validation data
                        validation_data = self._extract_validation_data(mb_data)
                        
                        # Step 6: Store research data (using database manager)
                        from .database_manager import SoundToolsDatabase
                        db = SoundToolsDatabase()
                        
                        success = db.store_research_data(
                            fingerprint=fingerprint,
                            ml_data=ml_analysis,
                            validation_data=validation_data
                        )
                        
                        if success:
                            logger.info(f"✅ Background research completed for {fingerprint[:8]}")
                            return True
                        else:
                            logger.warning(f"⚠️  Research data storage failed for {fingerprint[:8]}")
                            return False
                    else:
                        logger.debug(f"🔄 No recordings found for {fingerprint[:8]}")
                        return False
                else:
                    logger.debug(f"🔄 No AcoustID matches for {fingerprint[:8]}")
                    return False
                    
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            logger.error(f"❌ Background research failed for {fingerprint[:8]}: {e}")
            return False
    
    def _extract_validation_data(self, mb_data: Dict) -> Dict[str, Any]:
        """Extract validation data from MusicBrainz response"""
        
        try:
            recording = mb_data.get('recording', {})
            
            # Extract basic metadata
            validation_data = {
                "title": recording.get('title', 'Unknown'),
                "artist": self._extract_artist_name(recording),
                "length": recording.get('length'),  # Duration in milliseconds
                "musicbrainz_id": recording.get('id'),
                "tags": self._extract_tags(recording),
                "validation_source": "musicbrainz"
            }
            
            # Extract any key/tempo information if available
            tags = validation_data.get("tags", {})
            
            # Some tracks have key information in tags
            if "key" in tags:
                validation_data["known_key"] = tags["key"]
            
            return validation_data
            
        except Exception as e:
            logger.error(f"❌ Validation data extraction failed: {e}")
            return {"validation_source": "musicbrainz", "error": str(e)}
    
    def _extract_artist_name(self, recording: Dict) -> str:
        """Extract primary artist name from recording data"""
        try:
            artist_credits = recording.get('artist-credit', [])
            if artist_credits and len(artist_credits) > 0:
                return artist_credits[0].get('artist', {}).get('name', 'Unknown Artist')
            else:
                return 'Unknown Artist'
        except:
            return 'Unknown Artist'
    
    def _extract_tags(self, recording: Dict) -> Dict[str, Any]:
        """Extract tags from recording data"""
        try:
            tags = recording.get('tag-list', [])
            tag_dict = {}
            
            for tag in tags:
                tag_name = tag.get('name', '').lower()
                tag_count = tag.get('count', 0)
                tag_dict[tag_name] = tag_count
            
            return tag_dict
        except:
            return {}
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of MusicBrainz services"""
        return {
            "acoustid_available": self.acoustid_available,
            "musicbrainz_available": self.musicbrainz_available,
            "research_enabled": self.acoustid_available and self.musicbrainz_available
        }