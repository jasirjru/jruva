import type { NextConfig } from "next";
// Verified production build

const rawBackendUrl = (process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8001").replace(/\/$/, "");
const backendApiUrl = rawBackendUrl.endsWith("/api/v1") ? rawBackendUrl : `${rawBackendUrl}/api/v1`;

const nextConfig: NextConfig = {
  ...(process.env.STANDALONE ? { output: "standalone" } : {}),
  allowedDevOrigins: ["127.0.0.1", "localhost", "*.trycloudflare.com", "*.loca.lt", "*.ngrok-free.app"],
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${backendApiUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
