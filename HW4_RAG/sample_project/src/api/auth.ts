import { login } from "../auth/login";

export async function postLogin(username: string, password: string) {
  return login(username, password);
}
