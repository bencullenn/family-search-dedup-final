"use client";

import * as React from "react";
import Image from "next/image";
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "./button";

export interface StackProps {
  key: number;
  photos: {
    id: number;
    image: string;
    title: string;
    flag_status?: string;
    flagged_by?: number;
  }[];
  className?: string;
  selectedUser: { id: number; name: string } | null;
  onImagesUpdated?: () => void;
}

const Stack = React.forwardRef<HTMLDivElement, StackProps>(
  ({ photos, className, selectedUser, onImagesUpdated }, ref) => {
    // Filter out photos that are flagged by the current user
    const visiblePhotos = photos.filter(
      (photo) =>
        !(
          photo.flag_status === "FLAGGED" &&
          photo.flagged_by === selectedUser?.id
        )
    );

    const [currentIndex, setCurrentIndex] = React.useState(0);
    const [isModalOpen, setIsModalOpen] = React.useState(false);
    const [isSelectingNonDuplicates, setIsSelectingNonDuplicates] =
      React.useState(false);
    const [selectedPhotos, setSelectedPhotos] = React.useState<Set<number>>(
      new Set()
    );
    const [showSuccessMessage, setShowSuccessMessage] = React.useState(false);

    // Preload images
    React.useEffect(() => {
      visiblePhotos.forEach((photo) => {
        const img = new window.Image();
        img.src = photo.image;
      });
    }, [visiblePhotos]);

    // Handle keyboard navigation
    React.useEffect(() => {
      const handleKeyDown = (e: KeyboardEvent) => {
        if (!isModalOpen) return;

        if (e.key === "ArrowLeft") {
          handlePrev();
        } else if (e.key === "ArrowRight") {
          handleNext();
        }
      };

      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }, [isModalOpen, currentIndex]);

    if (!visiblePhotos.length) return null;

    const handleNext = () => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % visiblePhotos.length);
    };

    const handlePrev = () => {
      setCurrentIndex((prevIndex) =>
        prevIndex === 0 ? visiblePhotos.length - 1 : prevIndex - 1
      );
    };

    const { image, title } = visiblePhotos[currentIndex] || visiblePhotos[0];

    const serverUrl = "http://127.0.0.1:8000";
    const togglePhotoSelection = (index: number) => {
      setSelectedPhotos((prev) => {
        const newSet = new Set(prev);
        if (newSet.has(index)) {
          newSet.delete(index);
        } else {
          newSet.add(index);
        }
        return newSet;
      });
    };

    const handleSubmitNonDuplicates = async () => {
      if (!selectedUser) {
        console.error("No user selected");
        return;
      }

      try {
        // Flag each selected photo
        const promises = Array.from(selectedPhotos).map(async (index) => {
          const photo = visiblePhotos[index];
          const formData = new FormData();
          formData.append("image_id", photo.id.toString());
          formData.append("user_id", selectedUser.id.toString());

          const response = await fetch(`${serverUrl}/api/flag-image`, {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`Failed to flag image ${photo.id}`);
          }
        });

        await Promise.all(promises);
        console.log("Successfully flagged images:", Array.from(selectedPhotos));

        setIsSelectingNonDuplicates(false);
        setShowSuccessMessage(true);
        setSelectedPhotos(new Set());

        // Call the callback to refresh images
        onImagesUpdated?.();

        // Hide success message and close modal after delay
        setTimeout(() => {
          setShowSuccessMessage(false);
          setIsModalOpen(false);
        }, 3000);
      } catch (error) {
        console.error("Error flagging images:", error);
      }
    };

    return (
      <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
        <DialogTrigger asChild>
          <div
            ref={ref}
            className={cn(
              "bg-white rounded-lg shadow overflow-hidden relative cursor-pointer group",
              className
            )}
            onClick={() => setIsModalOpen(true)}
          >
            <div className="relative aspect-square">
              <Image
                src={image}
                alt={title}
                fill
                className="object-cover transition-opacity duration-200"
                priority={true}
                loading="eager"
              />
              {/* Stack indicator badge */}
              {visiblePhotos.length > 1 && (
                <div className="absolute top-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded-full">
                  {currentIndex + 1} / {visiblePhotos.length}
                </div>
              )}
              {/* Always show navigation buttons if there's more than one photo */}
              {visiblePhotos.length > 1 && (
                <div className="absolute inset-0 flex items-center justify-between px-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handlePrev();
                    }}
                    className="bg-black/50 hover:bg-black/70 text-white p-2 rounded-full shadow transition-colors"
                    aria-label="Previous image"
                  >
                    <ChevronLeft size={20} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleNext();
                    }}
                    className="bg-black/50 hover:bg-black/70 text-white p-2 rounded-full shadow transition-colors"
                    aria-label="Next image"
                  >
                    <ChevronRight size={20} />
                  </button>
                </div>
              )}
              {/* Visual indicator for stacks */}
              {visiblePhotos.length > 1 && (
                <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex gap-1">
                  {visiblePhotos.map((_, index) => (
                    <div
                      key={index}
                      className={cn(
                        "w-1.5 h-1.5 rounded-full bg-white/50",
                        index === currentIndex && "bg-white"
                      )}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        </DialogTrigger>

        <DialogContent className="max-w-3xl p-0 max-h-[90vh] overflow-y-auto">
          <div className="sticky top-0 bg-white z-10">
            <div className="p-6 pb-4 border-b">
              <div className="flex items-center justify-between mb-4">
                <DialogTitle>Photo Gallery</DialogTitle>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsModalOpen(false)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>

              {showSuccessMessage ? (
                <div className="bg-green-50 p-4 rounded-lg text-center">
                  <p className="text-green-700 font-medium mb-1">
                    Thank you for your help!
                  </p>
                  <p className="text-green-600 text-sm">
                    The selected photos have been flagged for review. Another
                    user in your tree will verify they don&apos;t belong in this
                    stack.
                  </p>
                </div>
              ) : !isSelectingNonDuplicates ? (
                <div className="flex items-center justify-center">
                  <Button
                    variant="outline"
                    onClick={() => setIsSelectingNonDuplicates(true)}
                    className="text-blue-600 hover:text-blue-700"
                  >
                    See some photos that are not duplicates?
                  </Button>
                </div>
              ) : (
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-blue-700 mb-2">
                    Select photos that you think are not duplicates of each
                    other.
                  </p>
                  <div className="flex justify-end gap-2">
                    <Button
                      variant="outline"
                      onClick={() => {
                        setIsSelectingNonDuplicates(false);
                        setSelectedPhotos(new Set());
                      }}
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={handleSubmitNonDuplicates}
                      disabled={selectedPhotos.size === 0}
                    >
                      Submit Selection
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="p-6 pt-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            {visiblePhotos.map((photo, index) => (
              <div
                key={index}
                className={cn(
                  "relative aspect-square rounded overflow-hidden cursor-pointer group",
                  isSelectingNonDuplicates &&
                    selectedPhotos.has(index) &&
                    "ring-2 ring-blue-500"
                )}
                onClick={() =>
                  isSelectingNonDuplicates && togglePhotoSelection(index)
                }
              >
                <Image
                  src={photo.image}
                  alt={photo.title}
                  fill
                  className="object-cover"
                  priority={index < 6}
                  loading={index < 6 ? "eager" : "lazy"}
                />
                {isSelectingNonDuplicates && (
                  <div
                    className={cn(
                      "absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-all",
                      selectedPhotos.has(index) && "bg-blue-500 bg-opacity-20"
                    )}
                  >
                    {selectedPhotos.has(index) && (
                      <div className="absolute top-2 right-2 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white">
                        ✓
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    );
  }
);

Stack.displayName = "Stack";

export { Stack };
