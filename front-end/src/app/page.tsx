"use client";

import { useEffect, useState, useCallback } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Stack } from "@/components/ui/stack";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { UploadDialog } from "@/components/upload-dialog";
import { ReviewDialog } from "@/components/review-dialog";

import {
  TreesIcon as Tree,
  Users,
  Star,
  Filter,
  Settings,
  Menu,
  ChevronDown,
  MapPinIcon,
  Globe,
  CircleHelp,
  MessagesSquare,
  Bell,
  FileStack,
  Loader2,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";

interface Memory {
  id: number;
  image: string;
  title: string;
  flag_status?: string;
  flagged_by?: number;
}

interface MemoryGroup {
  photos: Memory[];
}

export interface Person {
  id: number;
  created_at: string;
  first_name: string;
  last_name: string;
  folder: string;
}

interface User {
  id: number;
  name: string;
  created_at: string;
}

// This is the url when running in dev mode
const serverUrl = "http://127.0.0.1:8000";

// This is the url when running in production mode
//const serverUrl = "http://0.0.0.0:8000";
export default function Page() {
  const [memoryImages, setMemoryImages] = useState<MemoryGroup[]>();
  const [people, setPeople] = useState<Person[]>([]);
  const [selectedPerson, setSelectedPerson] = useState<Person | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [isGeneratingStacks, setIsGeneratingStacks] = useState<boolean>(false);
  const [isLoadingImages, setIsLoadingImages] = useState<boolean>(false);

  const fetchMemories = useCallback(async () => {
    if (!selectedPerson || !selectedUser) {
      console.log("No person or user selected, skipping fetch", {
        selectedPerson: selectedPerson?.id,
        selectedUser: selectedUser?.id,
      });
      return;
    }

    setIsLoadingImages(true);
    try {
      console.log("=== Starting fetchMemories ===");
      console.log("Selected person:", selectedPerson);
      console.log("Selected user:", selectedUser);

      const url = new URL(`${serverUrl}/api/person-images`);
      url.searchParams.append("person_id", selectedPerson.id.toString());
      url.searchParams.append("user_id", selectedUser.id.toString());

      console.log("Fetching from URL:", url.toString());

      const response = await fetch(url.toString());
      console.log("Response status:", response.status);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Raw data received:", JSON.stringify(data, null, 2));

      const images: MemoryGroup[] = [];
      for (const stack of data) {
        console.log("Processing stack:", JSON.stringify(stack, null, 2));
        const group: Memory[] = [];
        for (const img of stack) {
          console.log("Processing image:", img);
          group.push({
            id: parseInt(img.id.toString()),
            image: img.url,
            title: "",
            flag_status: img.flag_status,
            flagged_by: img.flagged_by,
          });
        }
        if (group.length > 0) {
          images.push({ photos: group });
        }
      }

      console.log("Final processed images:", JSON.stringify(images, null, 2));
      setMemoryImages(images);
    } catch (error) {
      console.error("Error fetching memories:", error);
      setMemoryImages([]);
    } finally {
      setIsLoadingImages(false);
    }
  }, [selectedPerson, selectedUser]);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await fetch(`${serverUrl}/api/users`);
        if (!response.ok) {
          throw new Error(`Error: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        if (data.length > 0) {
          setSelectedUser(data[0]);
        }
        setUsers(data);
      } catch (err) {
        console.error("Failed to fetch users:", err);
      }
    };
    fetchUsers();
  }, []);

  useEffect(() => {
    const fetchNames = async () => {
      try {
        const response = await fetch(`${serverUrl}/api/people`);
        if (!response.ok) {
          throw new Error(`Error: ${response.status} ${response.statusText}`);
        }
        const data: Person[] = await response.json();
        if (data.length > 0) {
          setSelectedPerson(data[0]);
        }
        setPeople(data);
      } catch (err) {
        console.error(err);
      }
    };
    fetchNames();
  }, []);

  const refreshMemories = async () => {
    // Refresh the memories list after upload
    await fetchMemories();
  };

  useEffect(() => {
    if (selectedPerson && selectedUser) {
      fetchMemories();
    }
  }, [selectedPerson, selectedUser, fetchMemories]);

  const handlePersonChange = (person: Person) => {
    setMemoryImages(undefined);
    setSelectedPerson(person);
  };

  const handleUserChange = (user: User) => {
    setMemoryImages(undefined);
    setSelectedUser(user);
  };

  const generateStacks = async () => {
    if (!selectedPerson) {
      return;
    }

    setIsGeneratingStacks(true);
    setMemoryImages(undefined);

    try {
      console.log(
        "Generating stacks for person",
        selectedPerson.first_name + " " + selectedPerson.last_name
      );

      const formData = new FormData();
      formData.append("person_id", selectedPerson?.id.toString());
      formData.append("match_threshold", "0.9");
      formData.append("recompute", "true");
      const response = await fetch(`${serverUrl}/api/create-stacks`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));

      await fetchMemories();
    } catch (error) {
      console.error("Error generating stacks:", error);
      setMemoryImages([]);
    } finally {
      setIsGeneratingStacks(false);
    }
  };

  return (
    <>
      {/* Top Navigation */}
      <nav className="border-b">
        <div className="max-w-7xl mx-auto px-4 flex items-center justify-between h-16">
          <div className="flex items-center gap-8">
            <Image
              src="/familysearch-tree.svg"
              alt="FamilySearch"
              width={120}
              height={40}
              className="h-10 w-auto"
            />
            <div className="flex gap-4">
              <a href="#" className="text-gray-700 hover:text-gray-900">
                <span className="hidden md:inline text-sm">Family Tree</span>
              </a>
              <a href="#" className="text-gray-700 hover:text-gray-900">
                <span className="hidden md:inline text-sm">Search</span>
              </a>
              <a href="#" className="text-gray-700 hover:text-gray-900">
                <span className="hidden md:inline text-sm">Memories</span>
              </a>
              <a href="#" className="text-gray-700 hover:text-gray-900">
                <span className="hidden md:inline text-sm">Get Involved</span>
              </a>
              <a href="#" className="text-gray-700 hover:text-gray-900">
                <span className="hidden md:inline text-sm">Activities</span>
              </a>
              <a href="#" className="text-gray-700 hover:text-gray-900">
                <span className="hidden md:inline text-sm">Temple</span>
              </a>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              className="hidden md:inline-flex"
            >
              <MapPinIcon className="h-6 w-6" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="hidden md:inline-flex"
            >
              <Globe className="h-6 w-6" />
            </Button>
            <Button variant="ghost" size="icon">
              <CircleHelp className="h-6 w-6" />
            </Button>
            <Button variant="ghost" size="icon">
              <MessagesSquare className="h-6 w-6" />
            </Button>
            <Button variant="ghost" size="icon">
              <div className="relative">
                <Bell className="h-6 w-6" />
                <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full" />
              </div>
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger className="hidden md:flex items-center gap-2 text-sm text-gray-700">
                {selectedUser?.name || "Select User"}
                <ChevronDown className="h-4 w-4" />
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                {users.map((user) => (
                  <DropdownMenuItem
                    key={user.id}
                    onClick={() => handleUserChange(user)}
                  >
                    {user.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <Button variant="ghost" size="icon" className="md:hidden">
              <Menu className="h-6 w-6" />
            </Button>
          </div>
        </div>
      </nav>

      {/* Secondary Navigation */}
      <nav className="border-b bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 flex items-center justify-between h-12">
          <div className="flex gap-6">
            {["Overview", "Tree", "Recents", "Find"].map((item) => (
              <a
                key={item}
                href="#"
                className="text-gray-700 hover:text-gray-900 text-sm"
              >
                {item}
              </a>
            ))}
            <DropdownMenu>
              <DropdownMenuTrigger className="flex items-center gap-1 text-sm text-gray-700 hover:text-gray-900">
                More <ChevronDown className="h-4 w-4" />
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                {[
                  "Following",
                  "Private People",
                  "My Contributions",
                  "Family Groups",
                ].map((item) => (
                  <DropdownMenuItem key={item}>{item}</DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              className="gap-2 hidden sm:inline-flex"
            >
              <Users className="w-4 h-4" />
              Help Others
            </Button>
            <Button variant="ghost" size="sm">
              <Settings className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header Section */}
        <div className="flex items-start gap-6 mb-8">
          <div className="relative w-24 h-24 rounded-full overflow-hidden border-2 border-gray-200">
            <Image
              src="/profile.png"
              alt="Profile"
              fill
              className="object-cover"
            />
          </div>
          <div className="flex-1">
            <DropdownMenu>
              <DropdownMenuTrigger className="flex items-center text-3xl font-semibold gap-1">
                {selectedPerson ? (
                  `${selectedPerson.first_name} ${selectedPerson.last_name}`
                ) : (
                  <div className="bg-gray-200 rounded animate-pulse w-80 h-10" />
                )}
                <ChevronDown className="h-5 w-5" />
              </DropdownMenuTrigger>
              <DropdownMenuContent className="max-h-60 overflow-y-auto">
                {people.map((person, index) => (
                  <DropdownMenuItem
                    key={index}
                    onClick={() => handlePersonChange(person)}
                  >
                    {person?.first_name + " " + person?.last_name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <p className="text-gray-600 mb-4">1938 - 1999 • LXYZ-243</p>
            <div className="flex gap-4">
              <Button variant="outline" size="sm" className="gap-2">
                <Tree className="w-4 h-4" />
                VIEW TREE
              </Button>
              <Button variant="outline" size="sm" className="gap-2">
                <Users className="w-4 h-4" />
                VIEW RELATIONSHIP
              </Button>
              <Button variant="outline" size="sm" className="gap-2">
                <Star className="w-4 h-4" />
                FOLLOW
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="gap-2"
                onClick={generateStacks}
                disabled={isGeneratingStacks || !selectedPerson}
              >
                {isGeneratingStacks ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <FileStack className="w-4 h-4" />
                    Generate Stacks
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <Tabs defaultValue="memories" className="mb-6">
          <TabsList className="w-full justify-start h-auto p-0 bg-transparent border-b">
            <TabsTrigger
              value="about"
              className="px-4 py-2 data-[state=active]:border-b-2 data-[state=active]:border-blue-600 rounded-none"
            >
              About
            </TabsTrigger>
            <TabsTrigger
              value="details"
              className="px-4 py-2 data-[state=active]:border-b-2 data-[state=active]:border-blue-600 rounded-none"
            >
              Details
            </TabsTrigger>
            <TabsTrigger
              value="sources"
              className="px-4 py-2 data-[state=active]:border-b-2 data-[state=active]:border-blue-600 rounded-none"
            >
              Sources (11)
            </TabsTrigger>
            <TabsTrigger
              value="collaborate"
              className="px-4 py-2 data-[state=active]:border-b-2 data-[state=active]:border-blue-600 rounded-none"
            >
              Collaborate (0)
            </TabsTrigger>
            <TabsTrigger
              value="memories"
              className="px-4 py-2 data-[state=active]:border-b-2 data-[state=active]:border-blue-600 rounded-none"
            >
              Memories (10)
            </TabsTrigger>
            <TabsTrigger
              value="timeline"
              className="px-4 py-2 data-[state=active]:border-b-2 data-[state=active]:border-blue-600 rounded-none"
            >
              Time Line
            </TabsTrigger>
            <TabsTrigger
              value="ordinances"
              className="px-4 py-2 data-[state=active]:border-b-2 data-[state=active]:border-blue-600 rounded-none"
            >
              <span className="flex items-center gap-2">
                <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center">
                  <Users className="w-3 h-3 text-white" />
                </div>
                Ordinances
              </span>
            </TabsTrigger>
          </TabsList>
        </Tabs>

        {/* Action Bar */}
        <div className="flex gap-4 mb-6">
          <UploadDialog
            person={selectedPerson!}
            onUploadSuccess={refreshMemories}
          />
          <ReviewDialog person={selectedPerson!} selectedUser={selectedUser} />
          <Button variant="outline" className="gap-2">
            <Filter className="w-4 h-4" />
            FILTER
          </Button>
          <div className="flex-1">
            <Input placeholder="Find Memories" className="max-w-md" />
          </div>
        </div>

        {/* Memories Grid with loading state */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {memoryImages === undefined || isLoadingImages ? (
            // Loading state
            Array.from({ length: 8 }).map((_, index) => (
              <div key={index} className="animate-pulse">
                <div className="bg-gray-200 rounded-lg h-48 mb-2"></div>
                <div className="bg-gray-200 h-4 w-3/4 rounded"></div>
              </div>
            ))
          ) : memoryImages.length > 0 ? (
            memoryImages.map((memoryGroup, index) => (
              <Stack
                key={index}
                photos={memoryGroup.photos}
                selectedUser={selectedUser}
                onImagesUpdated={fetchMemories}
              />
            ))
          ) : (
            <p>No memories found.</p>
          )}
        </div>
      </div>
    </>
  );
}
