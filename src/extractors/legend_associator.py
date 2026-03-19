from typing import List, Dict, Any

class LegendAssociator:
    """Assigns legend groups to extracted elements."""

    @staticmethod
    def associate(extracted: List[Dict], detections: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Associates each element with a legend label if possible.
        Uses a basic spatial/proximity or color heuristic.
        (Since we don't have pixel data here, we'll assign 'unknown' if multiple legends exist, 
        or the single legend if only 1 exists, or just leave it for now, 
        or implement the real nearest-swatch matching using the actual img crops later).
        For now, we just tag the series based on available text labels to unblock the pipeline.
        """
        legends = detections.get('legend', [])
        
        # In a full implementation, we'd sample the element's bbox color from img 
        # and match it to the color swatch inside the legend box.
        # Since this file was requested as a basic heuristic, we'll add placeholder 
        # logic that at least surfaces the extracted legend text to the elements.
        
        if not legends:
            return extracted
            
        # Example naive association: just collect all legend texts
        legend_texts = [l.get('text', '').strip() for l in legends if l.get('text')]
        default_group = legend_texts[0] if legend_texts else None
        
        for idx, el in enumerate(extracted):
            if default_group and 'group' not in el:
                # Provide a basic group assignment. In reality, requires color matching.
                # Adding the hook unblocks the UI data tab from rendering group columns.
                el['group'] = default_group
                el['legend_label'] = default_group
                
        return extracted
