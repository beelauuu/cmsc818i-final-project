import yaml
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from pathlib import Path
import re

# Configuration
SEMGREP_DIR = "./data/semgrep-rules"
CWE_XML = "./data/cwe-outputs/cwec_v4.18.xml" 
OUTPUT_FILE = "./outputs/cwe_rule_pairs.json"

class CWERulePairExtractor:
    def __init__(self, semgrep_dir, cwe_xml):
        self.semgrep_dir = semgrep_dir
        self.cwe_xml = cwe_xml
        self.model = None
        
    def extract_cwe_tagged_rules(self):
        """Step 1: Extract rules that explicitly have CWE tags"""
        print("=" * 60)
        print("STEP 1: Extracting rules with explicit CWE tags")
        print("=" * 60)
        
        cwe_to_rules = defaultdict(list)
        total_files = 0
        tagged_rules = 0
        
        for rule_file in glob.glob(f"{self.semgrep_dir}/**/*.yaml", recursive=True):
            total_files += 1
            try:
                with open(rule_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    
                    if not data or 'rules' not in data:
                        continue
                    
                    for rule in data['rules']:
                        metadata = rule.get('metadata', {})
                        
                        # Look for CWE in various metadata fields
                        cwe = (metadata.get('cwe') or 
                            metadata.get('CWE') or
                            metadata.get('cwe-id') or
                            metadata.get('cwe_id'))
                        
                        if cwe:
                            # Handle both single CWE and list of CWEs
                            cwe_list = cwe if isinstance(cwe, list) else [cwe]
                            
                            for cwe_id in cwe_list:
                                # Normalize to "CWE-XXX" format
                                cwe_str = str(cwe_id)
                                
                                match = re.search(r'CWE-(\d+)', cwe_str, re.IGNORECASE)
                                if match:
                                    cwe_normalized = f"CWE-{match.group(1)}"
                                elif cwe_str.isdigit():
                                    cwe_normalized = f"CWE-{cwe_str}"
                                else:
                                    # Skip if we can't parse it
                                    continue
                                
                                cwe_to_rules[cwe_normalized].append({
                                    'rule_id': rule.get('id', 'unknown'),
                                    'message': rule.get('message', ''),
                                    'severity': rule.get('severity', ''),
                                    'languages': rule.get('languages', []),
                                    'file': rule_file,
                                    'full_rule': rule,
                                    'source': 'explicit_tag'
                                })
                                tagged_rules += 1
                                
            except Exception as e:
                # Skip files that can't be parsed
                pass
        
        print(f"Processed {total_files} rule files")
        print(f"Found {tagged_rules} rules with CWE tags")
        print(f"Covering {len(cwe_to_rules)} unique CWEs")
        print()
        
        return cwe_to_rules
    
    def extract_cwe_database(self):
        """Step 2: Extract CWE descriptions from XML database"""
        print("=" * 60)
        print("STEP 2: Loading CWE database")
        print("=" * 60)
        
        tree = ET.parse(self.cwe_xml)
        root = tree.getroot()
        
        # Handle XML namespace
        ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}
        
        cwes = {}
        for weakness in root.findall('.//cwe:Weakness', ns):
            cwe_id = f"CWE-{weakness.get('ID')}"
            name = weakness.get('Name', '')
            
            # Extract description
            desc_elem = weakness.find('.//cwe:Description', ns)
            description = ''
            if desc_elem is not None:
                # Get all text content
                description = ''.join(desc_elem.itertext()).strip()
            
            # Extract extended description if available
            ext_desc_elem = weakness.find('.//cwe:Extended_Description', ns)
            extended_description = ''
            if ext_desc_elem is not None:
                extended_description = ''.join(ext_desc_elem.itertext()).strip()
            
            # Combine for matching
            full_text = f"{name}. {description}"
            if extended_description:
                full_text += f" {extended_description}"
            
            cwes[cwe_id] = {
                'name': name,
                'description': description,
                'extended_description': extended_description,
                'text': full_text
            }
        
        print(f"Loaded {len(cwes)} CWEs from database")
        print()
        
        return cwes
    
    def extract_all_rules(self):
        """Extract all Semgrep rules for NLP matching"""
        print("=" * 60)
        print("STEP 3: Extracting all rules for NLP matching")
        print("=" * 60)
        
        rules = []
        
        for rule_file in glob.glob(f"{self.semgrep_dir}/**/*.yaml", recursive=True):
            try:
                with open(rule_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    
                    if not data or 'rules' not in data:
                        continue
                    
                    for rule in data['rules']:
                        rule_id = rule.get('id', 'unknown')
                        message = rule.get('message', '')
                        
                        # Only include rules with meaningful messages
                        if message and len(message) > 10:
                            rules.append({
                                'rule_id': rule_id,
                                'message': message,
                                'severity': rule.get('severity', ''),
                                'languages': rule.get('languages', []),
                                'file': rule_file,
                                'full_rule': rule
                            })
            except:
                pass
        
        print(f"Extracted {len(rules)} rules with messages")
        print()
        
        return rules
    
    def nlp_match_rules_to_cwes(self, cwes, rules, threshold=0.5):
        """Step 4: Use NLP to match rules without CWE tags"""
        print("=" * 60)
        print("STEP 4: NLP matching of untagged rules to CWEs")
        print("=" * 60)
        
        # Load embedding model
        print("Loading sentence transformer model...")
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Prepare texts
        cwe_ids = list(cwes.keys())
        cwe_texts = [cwes[cwe_id]['text'] for cwe_id in cwe_ids]
        rule_messages = [r['message'] for r in rules]
        
        print(f"Computing embeddings for {len(cwe_texts)} CWEs...")
        cwe_embeddings = self.model.encode(cwe_texts, show_progress_bar=True)
        
        print(f"Computing embeddings for {len(rule_messages)} rules...")
        rule_embeddings = self.model.encode(rule_messages, show_progress_bar=True)
        
        print("Computing similarity matrix...")
        similarity = cosine_similarity(rule_embeddings, cwe_embeddings)
        
        # Match each rule to best CWE
        matches = defaultdict(list)
        matched_count = 0
        
        for rule_idx, rule in enumerate(rules):
            best_cwe_idx = np.argmax(similarity[rule_idx])
            best_score = similarity[rule_idx][best_cwe_idx]
            
            if best_score >= threshold:
                best_cwe = cwe_ids[best_cwe_idx]
                matches[best_cwe].append({
                    **rule,
                    'confidence': float(best_score),
                    'source': 'nlp_match'
                })
                matched_count += 1
        
        print(f"Matched {matched_count} rules to {len(matches)} CWEs (threshold={threshold})")
        print()
        
        return matches
    
    def combine_and_deduplicate(self, tagged_pairs, nlp_pairs):
        """Step 5: Combine both sources and remove duplicates"""
        print("=" * 60)
        print("STEP 5: Combining and deduplicating")
        print("=" * 60)
        
        combined = defaultdict(list)
        
        # Add tagged rules first (higher priority)
        for cwe_id, rules in tagged_pairs.items():
            combined[cwe_id].extend(rules)
        
        # Add NLP-matched rules, avoiding duplicates
        for cwe_id, rules in nlp_pairs.items():
            existing_ids = {r['rule_id'] for r in combined[cwe_id]}
            
            for rule in rules:
                if rule['rule_id'] not in existing_ids:
                    combined[cwe_id].append(rule)
                    existing_ids.add(rule['rule_id'])
        
        print(f"Total CWEs with rules: {len(combined)}")
        print()
        
        return combined
    
    def enrich_with_cwe_info(self, pairs, cwes):
        """Step 6: Add CWE information to each pair"""
        print("=" * 60)
        print("STEP 6: Enriching with CWE metadata")
        print("=" * 60)
        
        enriched = {}
        
        for cwe_id, rules in pairs.items():
            enriched[cwe_id] = {
                'cwe_info': cwes.get(cwe_id, {
                    'name': 'Unknown',
                    'description': 'CWE not found in database',
                    'text': ''
                }),
                'rules': rules,
                'rule_count': len(rules),
                'tagged_count': len([r for r in rules if r.get('source') == 'explicit_tag']),
                'nlp_count': len([r for r in rules if r.get('source') == 'nlp_match'])
            }
        
        print(f"Enriched {len(enriched)} CWE-rule pairs")
        print()
        
        return enriched
    
    def run_pipeline(self, nlp_threshold=0.6):
        """Run the complete extraction pipeline"""
        print("\n" + "=" * 60)
        print("STARTING CWE-RULE PAIR EXTRACTION PIPELINE")
        print("=" * 60 + "\n")
        
        # Step 1: Extract explicitly tagged rules
        tagged_pairs = self.extract_cwe_tagged_rules()
        
        # Step 2: Load CWE database
        cwes = self.extract_cwe_database()
        
        # Step 3: Extract all rules
        all_rules = self.extract_all_rules()
        
        # Step 4: Filter out already-tagged rules for NLP matching
        tagged_rule_ids = set()
        for rules in tagged_pairs.values():
            for r in rules:
                tagged_rule_ids.add(r['rule_id'])
        
        untagged_rules = [r for r in all_rules if r['rule_id'] not in tagged_rule_ids]
        print(f"Untagged rules to match: {len(untagged_rules)}\n")
        
        # Step 5: NLP matching for untagged rules
        nlp_pairs = self.nlp_match_rules_to_cwes(cwes, untagged_rules, threshold=nlp_threshold)
        
        # Step 6: Combine and deduplicate
        combined_pairs = self.combine_and_deduplicate(tagged_pairs, nlp_pairs)
        
        # Step 7: Enrich with CWE info
        enriched_pairs = self.enrich_with_cwe_info(combined_pairs, cwes)
        
        return enriched_pairs
    
    def save_results(self, pairs, output_file):
        """Save results to JSON"""
        print("=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)
        
        # Create output directory if needed
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        total_rules = sum(p['rule_count'] for p in pairs.values())
        total_tagged = sum(p['tagged_count'] for p in pairs.values())
        total_nlp = sum(p['nlp_count'] for p in pairs.values())
        
        print(f"Total CWEs: {len(pairs)}")
        print(f"Total rules: {total_rules}")
        print(f"  - Explicitly tagged: {total_tagged}")
        print(f"  - NLP matched: {total_nlp}")
        
        # Top 10 CWEs by rule count
        top_cwes = sorted(pairs.items(), key=lambda x: x[1]['rule_count'], reverse=True)[:10]
        
        print("\nTop 10 CWEs by rule count:")
        for cwe_id, data in top_cwes:
            name = data['cwe_info'].get('name', 'Unknown')
            count = data['rule_count']
            tagged = data['tagged_count']
            nlp = data['nlp_count']
            print(f"  {cwe_id}: {count} rules ({tagged} tagged, {nlp} NLP) - {name}")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    extractor = CWERulePairExtractor(SEMGREP_DIR, CWE_XML)
    pairs = extractor.run_pipeline(nlp_threshold=0.6)
    extractor.save_results(pairs, OUTPUT_FILE)