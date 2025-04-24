"use client";

import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import Image from "next/image";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

interface ReviewDialogProps {
  person: { id: number; first_name: string; last_name: string };
  selectedUser: { id: number; name: string } | null;
}

interface FlaggedStack {
  stack_id: number;
  flagged_image_id: number;
  flagged_by: number;
  images: {
    id: number;
    url: string;
    is_flagged: boolean;
    flagged_by: number;
  }[];
}

export function ReviewDialog({ person, selectedUser }: ReviewDialogProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [flaggedStacks, setFlaggedStacks] = useState<FlaggedStack[]>([]);
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string>("");
  const serverUrl = "http://127.0.0.1:8000";

  const showSuccessMessage = (message: string) => {
    setSuccessMessage(message);
    setTimeout(() => setSuccessMessage(""), 3000);
  };

  const fetchFlaggedImages = async () => {
    if (!person || !selectedUser) {
      console.log("Missing person or user", { person, selectedUser });
      return;
    }

    try {
      setLoading(true);
      console.log("Fetching flagged images for:", {
        person_id: person.id,
        user_id: selectedUser.id,
      });

      const response = await fetch(
        `${serverUrl}/api/flagged-images?person_id=${person.id}&user_id=${selectedUser.id}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Received flagged images:", data);
      setFlaggedStacks(data);
    } catch (error) {
      console.error("Error fetching flagged images:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeepInStack = async (imageId: number) => {
    try {
      const formData = new FormData();
      formData.append("image_id", imageId.toString());

      const response = await fetch(`${serverUrl}/api/unflag-image`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Failed to unflag image ${imageId}`);
      }

      // Refresh the flagged images list
      await fetchFlaggedImages();
      showSuccessMessage("Image kept in stack and unflagged");
    } catch (error) {
      console.error("Error keeping image in stack:", error);
    }
  };

  const handleRemoveFromStack = async (imageId: number) => {
    try {
      const formData = new FormData();
      formData.append("image_id", imageId.toString());

      const response = await fetch(`${serverUrl}/api/remove-from-stack`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Failed to remove image ${imageId} from stack`);
      }

      // Refresh the flagged images list
      await fetchFlaggedImages();
      showSuccessMessage("Image removed from stack");
    } catch (error) {
      console.error("Error removing image from stack:", error);
    }
  };

  useEffect(() => {
    fetchFlaggedImages();
  }, [person?.id, selectedUser?.id]);

  if (loading) {
    return (
      <Button variant="outline" disabled>
        Loading...
      </Button>
    );
  }

  if (!flaggedStacks.length) {
    return null;
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="gap-2">
          Review Flagged Photos ({flaggedStacks.length})
        </Button>
      </DialogTrigger>

      <DialogContent className="max-w-3xl p-0 max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-white z-10">
          <div className="p-6 pb-4 border-b">
            <DialogTitle>Review Flagged Photos</DialogTitle>
            {successMessage && (
              <div className="mt-2 p-2 bg-green-50 text-green-700 rounded">
                {successMessage}
              </div>
            )}
            <p className="mt-2 text-sm text-gray-600">
              Other users have identified these photos as potential
              non-duplicates. Please review their selections.
            </p>
          </div>
        </div>

        <div className="p-6">
          {flaggedStacks.map((stack, stackIndex) => (
            <div key={stackIndex} className="mb-8 last:mb-0">
              <h3 className="text-sm font-medium mb-4">
                Stack {stack.stack_id} - Flagged by User {stack.flagged_by}
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {stack.images.map((image, imageIndex) => (
                  <div
                    key={imageIndex}
                    className={cn(
                      "relative aspect-square rounded-lg overflow-hidden",
                      image.is_flagged && "ring-2 ring-red-500"
                    )}
                  >
                    <Image
                      src={image.url}
                      alt={`Stack ${stack.stack_id} Image ${imageIndex + 1}`}
                      fill
                      className="object-cover"
                    />
                    {image.is_flagged && (
                      <div className="absolute top-2 right-2 bg-red-500 text-white text-xs px-2 py-1 rounded-full">
                        Flagged
                      </div>
                    )}
                  </div>
                ))}
              </div>
              <div className="mt-4 flex justify-end gap-2">
                <Button
                  variant="outline"
                  onClick={() => handleKeepInStack(stack.flagged_image_id)}
                >
                  Keep in Stack
                </Button>
                <Button
                  variant="default"
                  onClick={() => handleRemoveFromStack(stack.flagged_image_id)}
                >
                  Remove from Stack
                </Button>
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
