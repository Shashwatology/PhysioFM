import os
import glob
import csv

class UBFCManager:
    """
    Manages the UBFC-rPPG dataset lifecycle:
    Validation -> Inventory -> Preprocessing split.
    """
    def __init__(self, data_root):
        self.data_root = data_root
        self.raw_dir = os.path.join(data_root, 'raw')
        self.meta_dir = os.path.join(data_root, 'metadata')
        
    def inventory_raw_data(self):
        """
        Scans the raw directory for subject folders, verifies contents,
        and generates an inventory report.
        """
        print(f"Scanning {self.raw_dir} for UBFC-rPPG subjects...")
        subjects = glob.glob(os.path.join(self.raw_dir, 'subject*'))
        
        inventory = []
        valid_count = 0
        for subj_path in subjects:
            subj_id = os.path.basename(subj_path)
            vid_path = os.path.join(subj_path, 'vid.avi')
            gt_path = os.path.join(subj_path, 'ground_truth.txt')
            
            has_vid = os.path.exists(vid_path)
            has_gt = os.path.exists(gt_path)
            valid = has_vid and has_gt
            if valid:
                valid_count += 1
                
            inventory.append({
                'Subject_ID': subj_id,
                'Has_Video': has_vid,
                'Has_GroundTruth': has_gt,
                'Valid': valid
            })
            
        if len(inventory) > 0:
            print(f"Found {len(inventory)} subjects. {valid_count} are fully valid.")
            
            # Save metadata
            os.makedirs(self.meta_dir, exist_ok=True)
            csv_path = os.path.join(self.meta_dir, 'ubfc_inventory.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['Subject_ID', 'Has_Video', 'Has_GroundTruth', 'Valid']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in inventory:
                    writer.writerow(row)
            print(f"Inventory saved to {csv_path}")
        else:
            print("\n[!] No subjects found in raw directory.")
            print("To proceed, please download the UBFC-rPPG dataset and extract the 'subjectX' folders directly into:")
            print(f"-> {self.raw_dir}")

if __name__ == "__main__":
    # Base path relative to this script running from root
    manager = UBFCManager(data_root=r"datasets\UBFC-rPPG")
    manager.inventory_raw_data()

