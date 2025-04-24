"use client";

import * as React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import Image from "next/image";

export interface DuplicateGroup {
  id: number;
  photos: { url: string; title?: string }[];
  selected: boolean;
}

interface DuplicateDetectionModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  uploadedImage: { url: string; title?: string } | null;
  duplicateGroups: DuplicateGroup[];
  onConfirm: (selectedGroups: DuplicateGroup[]) => void;
}

export function DuplicateDetectionModal({
  open,
  onOpenChange,
  uploadedImage,
  duplicateGroups: initialGroups,
  onConfirm,
}: DuplicateDetectionModalProps) {
  const [duplicateGroups, setDuplicateGroups] =
    React.useState<DuplicateGroup[]>(initialGroups);

  // Add this useEffect to update groups when initialGroups changes
  React.useEffect(() => {
    console.log("DuplicateDetectionModal received groups:", initialGroups); // Add this log
    setDuplicateGroups(initialGroups);
  }, [initialGroups]);

  const handleGroupSelect = (groupId: number) => {
    setDuplicateGroups((groups) =>
      groups.map((group) =>
        group.id === groupId ? { ...group, selected: !group.selected } : group
      )
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Possible Duplicate Images Found</DialogTitle>
          <DialogDescription>
            We&apos;ve detected some similar images in your collection. Please
            select any groups that contain duplicate images of your upload and
            then click Submit.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-6">
          {/* Uploaded Image */}
          <div className="border-b pb-4">
            <h3 className="font-medium mb-2">Your Upload</h3>
            {uploadedImage && (
              <div className="relative h-32 w-32">
                <Image
                  src={uploadedImage.url}
                  alt={uploadedImage.title || "Uploaded image"}
                  fill
                  className="object-cover rounded-lg"
                />
              </div>
            )}
          </div>

          {/* Potential Duplicates */}
          <div className="space-y-6">
            <h3 className="font-medium">Potential Matches</h3>
            {duplicateGroups.map((group) => (
              <div
                key={group.id}
                className="flex items-start gap-4 p-4 border rounded-lg"
              >
                <Checkbox
                  checked={group.selected}
                  onCheckedChange={() => handleGroupSelect(group.id)}
                />
                <div className="flex-1">
                  <p className="text-sm text-muted-foreground mb-2">
                    Group {group.id} • {group.photos.length} similar images
                  </p>
                  <div className="grid grid-cols-3 gap-2">
                    {group.photos.map((photo, photoIndex) => (
                      <div key={photoIndex} className="relative h-24 w-24">
                        <Image
                          src={photo.url}
                          alt={photo.title ? photo.title : ""}
                          fill
                          className="object-cover rounded-lg"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex justify-end mt-6">
          <Button onClick={() => onConfirm(duplicateGroups)}>Submit</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
