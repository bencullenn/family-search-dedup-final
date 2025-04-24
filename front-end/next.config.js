/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '3001',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'boiqldvzgaejmqxbsapi.supabase.co',
        pathname: '/**',
      },
    ],
  },
}

module.exports = nextConfig 