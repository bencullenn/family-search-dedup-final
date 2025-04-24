"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Upload, ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  DuplicateDetectionModal,
  DuplicateGroup,
} from "@/components/duplicate-detection-modal";
import { Person } from "@/app/page";
import { useState } from "react";

interface UploadDialogProps {
  onUploadSuccess: () => void;
  person: Person;
}

export function UploadDialog({ onUploadSuccess, person }: UploadDialogProps) {
  const [file, setFile] = React.useState<File | null>(null);
  const [title, setTitle] = React.useState("");
  const [isUploading, setIsUploading] = React.useState(false);
  const [open, setOpen] = React.useState(false);
  const [isDragging, setIsDragging] = React.useState(false);
  const [showDuplicates, setShowDuplicates] = useState(false);
  const [duplicateGroups, setDuplicateGroups] = useState<DuplicateGroup[]>([]);
  const [uploadedImagePreview, setUploadedImagePreview] = useState<{
    url: string;
    title?: string;
  } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    try {
      // First check for matches before uploading
      const matchFormData = new FormData();
      matchFormData.append("file", file);
      matchFormData.append("person_id", person.id.toString());
      matchFormData.append("match_threshold", "0.7");
      matchFormData.append("match_count", "10");

      console.log("Checking for matches with person_id:", person.id);

      const matchResponse = await fetch("http://127.0.0.1:8000/api/matches", {
        method: "POST",
        body: matchFormData,
      });

      if (!matchResponse.ok) {
        throw new Error(`Error checking matches: ${matchResponse.status}`);
      }

      const matches = await matchResponse.json();
      console.log("Raw matches response:", matches);

      // If we found matches, show the duplicate detection modal
      if (matches && matches.length > 0) {
        console.log("Found matches, processing...");

        const stackGroups = new Map();

        // Group matches by stack_id
        for (const match of matches) {
          console.log("Processing match:", match);
          const [title, url, stackId] = match;
          if (!stackGroups.has(stackId)) {
            stackGroups.set(stackId, {
              id: stackId,
              selected: false,
              photos: [],
            });
          }
          stackGroups.get(stackId).photos.push({ url, title });
        }

        const transformedGroups = Array.from(stackGroups.values());
        console.log("Transformed groups:", transformedGroups);

        // Create preview URL for uploaded image
        const imageUrl = URL.createObjectURL(file);
        setUploadedImagePreview({ url: imageUrl, title: title || file.name });
        setDuplicateGroups(transformedGroups);
        setShowDuplicates(true);
        setOpen(false);
        return;
      } else {
        console.log("No matches found, proceeding with normal upload");
      }

      // If no matches, proceed with normal upload
      await performUpload();
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Failed to upload image. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  // Separate the upload logic into its own function
  const performUpload = async (stack_id?: number) => {
    if (!file) return;

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("person", person.id.toString());
      formData.append("title", title || file.name);
      if (stack_id) {
        formData.append("stack_id", stack_id.toString());
      }

      const response = await fetch("http://127.0.0.1:8000/api/upload-image", {
        method: "POST",
        body: formData,
      });

      const responseData = await response.json();

      if (!response.ok) {
        throw new Error(responseData.detail || "Upload failed");
      }

      onUploadSuccess();
      setOpen(false);
      setFile(null);
      setTitle("");
    } catch (error) {
      console.error("Upload failed:", error);
      alert(error instanceof Error ? error.message : "Upload failed");
    }
  };

  const handleDuplicateConfirm = async (selectedGroups: DuplicateGroup[]) => {
    setShowDuplicates(false);

    // If no groups were selected, proceed with normal upload
    if (!selectedGroups.some((group) => group.selected)) {
      await performUpload();
      return;
    }

    // Get the first selected group's stack ID
    const selectedStack = selectedGroups.find((group) => group.selected)?.id;

    if (selectedStack) {
      await performUpload(selectedStack);
    }
  };

  return (
    <>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <Button className="gap-2">
            <Upload className="w-4 h-4" />
            UPLOAD MEMORY
          </Button>
        </DialogTrigger>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Upload Memory</DialogTitle>
            <DialogDescription>
              Upload a photo or document to add to this person&apos;s memories.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Input
                id="title"
                placeholder="Memory title (optional)"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>
            <div
              className={cn(
                "grid gap-2 place-items-center border-2 border-dashed rounded-lg p-4 transition-colors",
                isDragging ? "border-primary bg-primary/10" : "border-muted",
                "cursor-pointer"
              )}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => document.getElementById("file-upload")?.click()}
            >
              <input
                id="file-upload"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <div className="flex flex-col items-center gap-2 text-center">
                {file ? (
                  <>
                    <ImageIcon className="w-8 h-8 text-primary" />
                    <p className="text-sm font-medium">{file.name}</p>
                    <p className="text-xs text-muted-foreground">
                      Click or drag to replace
                    </p>
                  </>
                ) : (
                  <>
                    <Upload className="w-8 h-8 text-muted-foreground" />
                    <p className="text-sm font-medium">
                      Drag photo here or click to upload
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Supports: JPG, PNG, GIF
                    </p>
                  </>
                )}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button
              type="submit"
              onClick={handleUpload}
              disabled={!file || isUploading}
            >
              {isUploading ? "Uploading..." : "Upload"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <DuplicateDetectionModal
        open={showDuplicates}
        onOpenChange={setShowDuplicates}
        uploadedImage={uploadedImagePreview}
        duplicateGroups={duplicateGroups}
        onConfirm={handleDuplicateConfirm}
      />
    </>
  );
}
