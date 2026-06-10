export function login(username: string, password: string) {
  if (!username || !password) {
    throw new Error("missing credentials");
  }
  return { username, token: "demo-token" };
}
