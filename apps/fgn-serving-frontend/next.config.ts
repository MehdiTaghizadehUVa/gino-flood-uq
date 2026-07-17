import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    localPatterns: [
      {
        pathname: "/marketing/**"
      }
    ]
  },
  async redirects() {
    return [
      {
        source: "/runs/:runId",
        destination: "/demo/runs/:runId",
        permanent: true
      },
      {
        source: "/runs",
        destination: "/demo?workspace=runs",
        permanent: true
      }
    ];
  }
};

export default nextConfig;
